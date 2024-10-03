import sys
sys.path.append('../')
sys.path.append('../../ext/NeuralHaircut')
sys.path.append('../../ext/NeuralHaircut/k-diffusion')
import torch
import numpy as np
import cv2
import pickle as pkl
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import build_scaling_rotation
from gaussian_renderer import GaussianModel
import trimesh
from pytorch3d import utils
from pysdf import SDF
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from src.utils.geometry import face_vertices
from skimage.draw import polygon



def prune_points(gaussians, mask):
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._label = gaussians._label[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._orient_conf = gaussians._orient_conf[mask]
    return gaussians


def create_scalp_mask(scalp_mesh, scalp_uvs):
    img = np.zeros((256, 256, 1), 'uint8')
    
    for i in range(scalp_mesh.faces_packed().shape[0]):
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy()
        poly = 255/2 * (text + 1)
        rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
        img[rr,cc, :] = (255)

    scalp_mask = np.flip(img.transpose(1, 0, 2), axis=0)
    return scalp_mask


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_name', default='', type=str) 
    parser.add_argument('--iter', default=30_000, type=int) 
    parser.add_argument('--project_dir', default="", type=str)
    args = get_combined_args(parser)

    print(f'{args.flame_mesh_dir}/stage_3/mesh_final.obj')

    mesh = trimesh.load(f'{args.flame_mesh_dir}/stage_3/mesh_final.obj')

    mesh_head = load_objs_as_meshes([f'{args.flame_mesh_dir}/stage_3/mesh_final.obj'], device=args.device)

    scalp_vert_idx = torch.load(f'{args.project_dir}/data/new_scalp_vertex_idx.pth').long().cuda()
    scalp_faces = torch.load(f'{args.project_dir}/data/new_scalp_faces.pth')[None].cuda() 
    scalp_uvs = torch.load(f'{args.project_dir}/data/improved_neural_haircut_uvmap.pth')[None].cuda()

    # Convert the head mesh into a scalp mesh
    scalp_verts = mesh_head.verts_packed()[None, scalp_vert_idx]
    scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).cuda()

    postfix = '_cropped'
    suffix = ''
    gaussians = GaussianModel(3)
    path = os.path.join(args.model_path,
                        f"point_cloud{postfix}",
                        f"iteration_{args.iter}",
                        f"raw_{suffix}point_cloud.ply")
    gaussians.load_ply(path)
    
    M = build_scaling_rotation(gaussians.get_scaling * 3, gaussians._rotation)

    # Check which Gaussians intersect with scalp
    x = scalp_mesh.verts_packed().view(-1, 1, 3, 1) # N x 1 x 3 x 1
    head_mask = gaussians.get_label.squeeze() < 0.1
    mu = gaussians.get_xyz[head_mask].view(1, -1, 3, 1) # 1 x M x 3 x 1
    Cov = (M.transpose(1, 2) @ M)[head_mask]
    d = ((x - mu).transpose(-1, -2) @ torch.linalg.inv(Cov)[None] @ (x - mu)).squeeze()**0.5 <= 2
    vis_mask = d.sum(1) <= 10
    sorted_idx = torch.where(vis_mask)[0]

    # Cut new scalp
    a = np.array(sorted(sorted_idx.cpu()))
    b = np.arange(a.shape[0])
    d = dict(zip(a,b))

    sphere = utils.ico_sphere(level=0, device='cuda')
    verts = sphere.verts_list()[0]
    faces = sphere.faces_list()[0]
    N_vert = verts.shape[0]

    verts_all = (verts[None, :, None, :] @ M[:, None, :, :])[:, :, 0, :] + gaussians.get_xyz[:, None]
    faces_all = faces[None] + torch.arange(M.shape[0], device='cuda')[:, None, None] * verts.shape[0]

    verts_all = verts_all.view(-1, 3)
    faces_all = faces_all.view(-1, 3)
    
    f = SDF(mesh.vertices, mesh.faces)
    
    sdf = f(verts_all.detach().cpu().numpy())
    outside_mesh = torch.from_numpy(sdf < 0).view(gaussians.get_xyz.shape[0], N_vert).all(dim=1)
    outside_mesh = torch.logical_or(outside_mesh, gaussians.get_label.cpu().squeeze() <= 0.5)
    gaussians = prune_points(gaussians, outside_mesh)
    gaussians.save_ply(f"{args.model_path}/point_cloud_filtered/iteration_{args.iter}/{suffix}point_cloud.ply")