import os
import sys
from pyhocon import ConfigFactory
from pathlib import Path

from PIL import Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
import cv2

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.io import load_ply, save_ply, save_obj, load_objs_as_meshes

sys.path.append('../../ext/NeuralHaircut')
from src.models.dataset import Dataset, MonocularDataset
from src.utils.geometry import face_vertices
from NeuS.models.dataset import load_K_Rt_from_P

sys.path.append('../')
sys.path.append('../../ext/NeuralHaircut/k-diffusion')
from utils.general_utils import build_scaling_rotation
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

import argparse
import yaml
import pickle as pkl
from skimage.draw import polygon

from tqdm import tqdm


def create_scalp_mask(scalp_mesh, scalp_uvs):
    img = np.zeros((256, 256, 1), 'uint8')
    
    for i in range(scalp_mesh.faces_packed().shape[0]):
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy()
        poly = 255/2 * (text + 1)
        rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
        img[rr,cc, :] = (255)

    scalp_mask = np.flip(img.transpose(1, 0, 2), axis=0)
    return scalp_mask


def create_visibility_map(camera, rasterizer, mesh, head_mask):
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face
    packed_faces = mesh.faces_packed() 
    packed_verts = mesh.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]) 
    vertex_visibility_map_head = torch.zeros(packed_verts.shape[0]) 
    faces_visibility_map = torch.zeros(packed_faces.shape[0])
    faces_visibility_map_head = torch.zeros(packed_faces.shape[0])
    visible_faces = pix_to_face.unique()[1:] # not take -1
    pix_to_face_head = torch.where(head_mask[None, ..., None], pix_to_face, -1 * torch.ones_like(pix_to_face))
    visible_faces_head = pix_to_face_head.unique()[1:] # not take -1
    visible_verts_idx = packed_faces[visible_faces] 
    visible_verts_head_idx = packed_faces[visible_faces_head] 
    unique_visible_verts_idx = torch.unique(visible_verts_idx)
    unique_visible_verts_head_idx = torch.unique(visible_verts_head_idx)
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    vertex_visibility_map_head[unique_visible_verts_head_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    faces_visibility_map_head[torch.unique(visible_faces_head)] = 1.0
    pix_to_face_vis = pix_to_face_head.clone() >= 0
    return vertex_visibility_map, vertex_visibility_map_head, faces_visibility_map, faces_visibility_map_head, pix_to_face_vis


def check_visiblity_of_faces(cams, masks, meshRasterizer, full_mesh, flame_mesh_dir, prob_thr, n_views_thr):
    # collect visibility maps
    os.makedirs(f'{flame_mesh_dir}/scalp_data/vis', exist_ok=True)
    vis_maps = []
    vis_maps_head = []
    for cam in cams.keys():
        head_mask = masks[cam] >= 0.5
        v, vh, _, _, vis = create_visibility_map(cams[cam], meshRasterizer, full_mesh, head_mask)
        Image.fromarray((vis[0, ..., 0].cpu().numpy() * 255).astype('uint8')).save(f'{flame_mesh_dir}/scalp_data/vis/{cam}.jpg')
        vis_maps.append(v)
        vis_maps_head.append(vh)
    vis_maps = torch.stack(vis_maps).sum(0).float()
    vis_maps_head = torch.stack(vis_maps_head).sum(0).float()

    prob_vis_head = vis_maps_head / vis_maps # probability of faces belonging to visible head
    prob_hair = 1 - prob_vis_head
    vis_mask = torch.logical_or(prob_hair > prob_thr, vis_maps / len(cams.keys()) < n_views_thr)

    return vis_mask


def main(args):
    mesh_head = load_objs_as_meshes([f'{args.flame_mesh_dir}/stage_3/mesh_final.obj'], device=args.device)

    scalp_vert_idx = torch.load(f'{args.project_dir}/data/new_scalp_vertex_idx.pth').long().cuda()
    scalp_faces = torch.load(f'{args.project_dir}/data/new_scalp_faces.pth')[None].cuda() 
    scalp_uvs = torch.load(f'{args.project_dir}/data/improved_neural_haircut_uvmap.pth')[None].cuda()

    # Convert the head mesh into a scalp mesh
    scalp_verts = mesh_head.verts_packed()[None, scalp_vert_idx]
    scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).cuda()

    cams_all = pkl.load(open(args.cams_path, 'rb'))
    masks = {}
    cams = {}
    for k in cams_all.keys():
        mask_hair = cv2.dilate(cv2.imread(f'{args.data_dir}/masks_2/hair/{k}.png'), np.ones((5, 5))) / 255. >= 0.5
        mask = cv2.dilate(cv2.imread(f'{args.data_dir}/masks_2/body/{k}.png'), np.ones((5, 5))) / 255. >= 0.5
        mask_head = np.clip(mask.astype('float32') - mask_hair.astype('float32'), 0, 1)
        masks[k] = torch.from_numpy(mask_head)[:, :, 0].cuda()
        intrinsics, pose = load_K_Rt_from_P(None, cams_all[k].transpose(0, 1)[:3, :4].numpy())
        pose_inv = np.linalg.inv(pose)
        intrinsics_modified = intrinsics.copy()

        intrinsics_modified[0, 0] /= 2  # Halve fx
        intrinsics_modified[1, 1] /= 2  # Halve fy
        intrinsics_modified[0, 2] /= 2  # Halve cx
        intrinsics_modified[1, 2] /= 2  # Halve cy

        intrinsics_modified[0, 2] += 0.5  # Adjust cx
        intrinsics_modified[1, 2] += 0.5  # Adjust cy

        scale_y, scale_x = masks[k].shape

        # Create a scaling matrix
        scaling_matrix = np.array([[scale_x, 0, 0, 0],
                                   [0, scale_y, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        intrinsics_scaled = scaling_matrix @ intrinsics_modified

        size = torch.tensor([scale_y, scale_x]).to(args.device)

        raster_settings_mesh = RasterizationSettings(
            image_size=(scale_y, scale_x), 
            blur_radius=0.000, 
            faces_per_pixel=1)

        cams[k] = cameras_from_opencv_projection(
            camera_matrix=torch.from_numpy(intrinsics_scaled).float()[None].cuda(), 
            R=torch.from_numpy(pose_inv[:3, :3]).float()[None].cuda(),
            tvec=torch.from_numpy(pose_inv[:3, 3]).float()[None].cuda(),
            image_size=size[None].cuda()
        ).cuda()

    # init camera
    R = torch.ones(1, 3, 3)
    t = torch.ones(1, 3)
    cam_intr = torch.ones(1, 4, 4)
    size = torch.tensor([scale_y, scale_x]).to(args.device)

    cam = cameras_from_opencv_projection(
        camera_matrix=cam_intr.cuda(), 
        R=R.cuda(),
        tvec=t.cuda(),
        image_size=size[None].cuda()
    ).cuda()

    # init mesh rasterization
    meshRasterizer = MeshRasterizer(cam, raster_settings_mesh)

    mesh_head.textures = TexturesVertex(verts_features=torch.ones_like(mesh_head.verts_packed()).float().cuda()[None])

    # join hair and bust mesh to handle occlusions
    full_mesh = mesh_head

    vis_vertex_mask = check_visiblity_of_faces(cams, masks, meshRasterizer, full_mesh, args.flame_mesh_dir, prob_thr=0.5, n_views_thr=0.1).cuda()

    # sorted_idx = torch.where(vis_mask | vis_vertex_mask.bool()[scalp_vert_idx])[0]
    # sorted_idx = vis_mask
    vis_vertex_mask_scalp = vis_vertex_mask.bool()[scalp_vert_idx]
    for i, j in zip([[327, 304, 286, 264, 247, 235], 
                     [236, 251, 271, 294, 309, 329], 
                     [336, 315, 298, 277, 253, 237], 
                     [238, 255, 284, 301, 324, 343],
                     [354, 330, 305, 285, 258, 239]], 
                    [[ 94, 114, 140, 156, 184, 201], 
                     [197, 179, 155, 138, 112,  92],
                     [ 87, 111, 136, 154, 171, 194],
                     [191, 165, 152, 125, 108,  84],
                     [ 79,  99, 118, 144, 159, 189]]):
        tmp = min(vis_vertex_mask_scalp[i].amin(), vis_vertex_mask_scalp[j].amin())
        vis_vertex_mask_scalp[i] = tmp
        vis_vertex_mask_scalp[j] = tmp
    
    for i, j in zip([414, 419, 425, 426, 422, 424, 421,
                     412, 417, 428, 433, 434, 429, 420, 410, 402,
                     403, 409, 415, 432, 437, 435, 423, 411, 398, 393, 387],
                    [ 17,  15,  12,  10,  13,   8,   5,
                      19,  16,   9,   3,   4,  11,  18,  23,  31,
                      27,  24,  20,   7,   0,   1,  22,  28,  36,  43,  47]):
        tmp = min(vis_vertex_mask_scalp[i], vis_vertex_mask_scalp[j])
        vis_vertex_mask_scalp[i] = tmp
        vis_vertex_mask_scalp[j] = tmp

    # vis_vertex_mask_scalp = torch.ones_like(vis_vertex_mask_scalp).bool()
    sorted_idx = torch.where(vis_vertex_mask_scalp)[0]

    # Cut new scalp
    a = np.array(sorted(sorted_idx.cpu()))
    b = np.arange(a.shape[0])
    d = dict(zip(a,b))

    full_scalp_list = sorted(sorted_idx)

    save_path = os.path.join(args.flame_mesh_dir, 'scalp_data')
    os.makedirs(save_path, exist_ok=True)

    faces_masked = []
    for face in scalp_mesh.faces_packed():
        if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
            faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))
    save_obj(os.path.join(save_path, 'scalp.obj'), scalp_mesh.verts_packed()[full_scalp_list], torch.stack(faces_masked))

    with open(os.path.join(save_path, 'cut_scalp_verts.pickle'), 'wb') as f:
        pkl.dump(list(torch.tensor(sorted_idx).detach().cpu().numpy()), f)
    
    # Create scalp mask for diffusion
    scalp_uvs = scalp_uvs[:, full_scalp_list]    
    scalp_mesh = load_objs_as_meshes([os.path.join(save_path, 'scalp.obj')], device=args.device)
    
    scalp_mask = create_scalp_mask(scalp_mesh, scalp_uvs)
    cv2.imwrite(os.path.join(save_path, 'dif_mask.png'), scalp_mask)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument('--project_dir', default="", type=str)
    parser.add_argument('--data_dir', default="", type=str)
    parser.add_argument('--flame_mesh_dir', default="", type=str)
    parser.add_argument('--cams_path', default="", type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = get_combined_args(parser)

    main(args)