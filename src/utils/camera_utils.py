#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import torch.nn.functional as F
import math
import os,sys,time
import collections
from easydict import EasyDict as edict
import tqdm



WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if hasattr(args, 'load_synthetic_rgba') and args.load_synthetic_rgba:
        basename = os.path.basename(cam_info.image_path).split('.')[0]
        resized_image_rgb = PILtoTorch(Image.open(f'{args.model_path}/train_cropped/ours_{args.iteration_data}/renders/{basename}.png'), resolution)
        resized_mask_body = PILtoTorch(Image.open(f'{args.model_path}/train_cropped/ours_{args.iteration_data}/head_masks/{basename}.png'), resolution)
        resized_mask_hair = PILtoTorch(Image.open(f'{args.model_path}/train_cropped/ours_{args.iteration_data}/hair_masks/{basename}.png'), resolution)
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        resized_mask_body = PILtoTorch(Image.open(cam_info.image_path.replace(f'images_2', f'masks_2/body')), resolution)
        resized_mask_hair = PILtoTorch(Image.open(cam_info.image_path.replace(f'images_2', f'masks_2/hair')), resolution)

    if hasattr(args, 'load_synthetic_geom') and args.load_synthetic_geom:
        basename = os.path.basename(cam_info.image_path).split('.')[0]
        resized_orient_angle = PILtoTorch(Image.open(f'{args.model_path}/train_cropped/ours_{args.iteration_data}/orients/{basename}.png'), resolution)
        resized_orient_conf = F.interpolate(torch.load(f'{args.model_path}/train_cropped/ours_{args.iteration_data}/orient_confs/{basename}.pth').float()[None], size=resolution[::-1], mode='bilinear')[0]
    else:
        resized_orient_angle = PILtoTorch(Image.open(cam_info.image_path.replace(f'images_2', f'orientations_2/angles').replace('png', 'png')), resolution, max_value=180.0) # [0, 1], where 1 stands for pi
        resized_orient_var = F.interpolate(torch.from_numpy(np.load(cam_info.image_path.replace(f'images_2', f'orientations_2/vars').replace('png', 'npy'))).float()[None, None], size=resolution[::-1], mode='bilinear')[0] / math.pi**2
        resized_orient_conf = 1 / (resized_orient_var ** 2 + 1e-7)

    gt_image = resized_image_rgb[:3, ...]
    gt_mask_body = resized_mask_body[:1, ...]
    gt_mask_hair = resized_mask_hair[:1, ...]
    if args.binarize_masks:
        gt_mask_body = (gt_mask_body >= 0.5).float()
        gt_mask_hair = (gt_mask_hair >= 0.5).float()
    gt_orient_angle = resized_orient_angle[:1, ...]
    gt_orient_conf = resized_orient_conf[:1, ...]
    gt_depth = torch.empty_like(gt_orient_conf)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, width=cam_info.width, height=cam_info.height,
                  image=gt_image, mask_body=gt_mask_body, mask_hair=gt_mask_hair, orient_angle=gt_orient_angle, orient_conf=gt_orient_conf, depth=gt_depth,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, white_background=args.white_background,
                  trainable_cameras=args.trainable_cameras, use_barf=args.use_barf, trainable_intrinsics=args.trainable_intrinsics)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm.tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry