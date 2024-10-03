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
from gaussian_renderer import render_hair
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import vis_orient
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel, GaussianModelCurves
import pickle as pkl
import yaml
import math
import shutil



def render_set(model_path, name, iteration, views, gaussians, gaussians_hair, pipeline, background, scene_suffix):
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
        output = render_hair(view, gaussians, gaussians_hair, pipeline, background)

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
def render_sets(dataset : ModelParams, optimizer: OptimizationParams, optimizer_hair, iteration : int, pipeline : PipelineParams, model_hair_path : str, pointcloud_path_head : str, checkpoint_hair : str, checkpoint_curves : str, skip_train : bool, skip_test : bool, scene_suffix : str):
    gaussians = GaussianModel(3)
    gaussians_hair = GaussianModelCurves(dataset.source_path, dataset.flame_mesh_dir, opt_hair, 3)
    scene = Scene(dataset, gaussians, pointcloud_path=pointcloud_path_head, load_iteration=-1)
    gaussians.training_setup(optimizer)
    
    # Initialize hair gaussians
    model_params, _ = torch.load(checkpoint_hair)
    gaussians_hair.create_from_pcd(dataset.source_path, model_params, 30_000, gaussians.spatial_lr_scale)
    model_params, _ = torch.load(checkpoint_curves)
    gaussians_hair.restore(model_params, optimizer_hair)
    gaussians_hair.use_sds = False

    gaussians_hair.initialize_gaussians_hair()

    # Precompute head gaussians
    gaussians.mask_precomp = gaussians.get_label[..., 0] < 0.5
    gaussians.xyz_precomp = gaussians.get_xyz[gaussians.mask_precomp].detach()
    gaussians.opacity_precomp = gaussians.get_opacity[gaussians.mask_precomp].detach()
    gaussians.scaling_precomp = gaussians.get_scaling[gaussians.mask_precomp].detach()
    gaussians.rotation_precomp = gaussians.get_rotation[gaussians.mask_precomp].detach()
    gaussians.cov3D_precomp = gaussians.get_covariance(1.0)[gaussians.mask_precomp].detach()
    gaussians.shs_view = gaussians.get_features[gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1)**2)

    bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 100] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(model_hair_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, gaussians_hair, pipeline, background, scene_suffix)

    if not skip_test:
        render_set(model_hair_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, gaussians_hair, pipeline, background, scene_suffix)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimizer = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--data_dir", type=str, default = None)
    parser.add_argument("--model_hair_path", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--checkpoint_hair", type=str, default = None)
    parser.add_argument("--checkpoint_curves", type=str, default = None)
    parser.add_argument("--scene_suffix", default="", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Configuration of hair strands
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        opt_hair = yaml.load(replaced_conf, Loader=yaml.Loader)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset = model.extract(args)
    if args.data_dir is not None:
        dataset.source_path = args.data_dir

    if args.max_frames > 200:
        # Split processing into multiple iterations
        max_frames = args.max_frames
        num_iterations = int(math.ceil(max_frames / 200))
        for i in range(num_iterations):
            dataset.max_frames = 200 if i < num_iterations - 1 else max_frames % 200
            dataset.offset = i * 200
            render_sets(dataset, optimizer.extract(args), opt_hair, args.iteration, pipeline.extract(args), args.model_hair_path, args.pointcloud_path_head, args.checkpoint_hair, args.checkpoint_curves, args.skip_train, args.skip_test, args.scene_suffix)
            
        # Clean extra files to conserve space
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/hair_masks')
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/head_masks')
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/orient_confs')
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/orient_confs_vis')
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/orients')
        shutil.rmtree(f'{args.model_hair_path}/train/ours_30000/orients_vis')