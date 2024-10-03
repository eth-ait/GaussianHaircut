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

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import vis_orient
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
import pickle as pkl



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_suffix):
    dir_name = f"{name}{scene_suffix}"
    render_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "renders")
    hair_mask_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "hair_masks")
    head_mask_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "head_masks")
    orient_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "orients")
    orient_vis_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "orients_vis")
    orient_conf_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "orient_confs")
    orient_conf_vis_path = os.path.join(model_path, dir_name, "ours_{}".format(iteration), "orient_confs_vis")

    makedirs(render_path, exist_ok=True)
    makedirs(hair_mask_path, exist_ok=True)
    makedirs(head_mask_path, exist_ok=True)
    makedirs(orient_path, exist_ok=True)
    makedirs(orient_vis_path, exist_ok=True)
    makedirs(orient_conf_path, exist_ok=True)
    makedirs(orient_conf_vis_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background)

        image = output["render"]
        hair_mask = output["mask"][:1]
        head_mask = output["mask"][1:]
        orient_angle = output["orient_angle"] * hair_mask
        orient_angle_vis = vis_orient(output["orient_angle"], hair_mask)
        orient_conf = output["orient_conf"] * hair_mask
        orient_conf_vis = (1 - 1 / (orient_conf + 1))
        orient_conf_vis = vis_orient(output["orient_angle"], orient_conf_vis)

        basename = os.path.basename(view.image_name).split('.')[0]
        torchvision.utils.save_image(image, os.path.join(render_path, basename + ".png"))
        torchvision.utils.save_image(hair_mask, os.path.join(hair_mask_path, basename + ".png"))
        torchvision.utils.save_image(head_mask, os.path.join(head_mask_path, basename + ".png"))
        torchvision.utils.save_image(orient_angle, os.path.join(orient_path, basename + ".png"))
        torchvision.utils.save_image(orient_angle_vis, os.path.join(orient_vis_path, basename + ".png"))
        torch.save(orient_conf, os.path.join(orient_conf_path, basename + ".pth"))
        torchvision.utils.save_image(orient_conf_vis, os.path.join(orient_conf_vis_path, basename + ".png"))

@torch.no_grad()
def render_sets(dataset : ModelParams, optimizer: OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, scene_suffix : str):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, scene_suffix=scene_suffix)
    gaussians.training_setup(optimizer)
    if dataset.trainable_cameras:
        print(f'Loading optimized cameras from iter {scene.loaded_iter}')
        params_cam_rotation, params_cam_translation, params_cam_fov = pkl.load(open(scene.model_path + "/cameras/" + str(scene.loaded_iter) + ".pkl", 'rb'))
        for k in scene.train_cameras.keys():
            for camera in scene.train_cameras[k]:
                if dataset.trainable_cameras:
                    camera._rotation_res.data = params_cam_rotation[camera.image_name]
                    camera._translation_res.data = params_cam_translation[camera.image_name]
                if dataset.trainable_intrinsics:
                    camera._fov_res.data = params_cam_fov[camera.image_name]
        projection_all = {}
        params_all = {}
        for camera in scene.train_cameras[1.0]:
            projection_all[camera.image_name] = camera.full_proj_transform.cpu()
            params_all[camera.image_name] = {
                'fx': fov2focal(camera.FoVx, camera.width).item(),
                'fy': fov2focal(camera.FoVy, camera.height).item(),
                'width': camera.width,
                'height': camera.height,
                'Rt': camera.world_view_transform.cpu().transpose(0, 1)
            }
        pkl.dump(projection_all, open(scene.model_path + "/cameras/" + str(scene.loaded_iter) + "_matrices.pkl", 'wb'))
        pkl.dump(params_all, open(scene.model_path + "/cameras/" + str(scene.loaded_iter) + "_params.pkl", 'wb'))

    bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 100] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene_suffix)

    if not skip_test:
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene_suffix)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimizer = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--scene_suffix", default="", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    if args.data_dir:
        dataset.source_path = args.data_dir

    render_sets(dataset, optimizer.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scene_suffix)