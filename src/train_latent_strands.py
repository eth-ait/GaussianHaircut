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

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_chamfer_utils import chamfer_distance
from utils.loss_utils import l1_loss, ce_loss, or_loss, ssim
from gaussian_renderer import render_hair, network_gui
import sys
import yaml
from scene import Scene, GaussianModel, GaussianModelHair
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, vis_orient
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle as pkl
from utils.general_utils import build_rotation
import time
from kaolin.metrics.pointcloud import sided_distance

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def training(dataset, opt, opt_hair, pipe, testing_iterations, saving_iterations, checkpoint_iterations, model_path_hair, pointcloud_path_head, checkpoint, checkpoint_hair, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, model_path_hair)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians_hair = GaussianModelHair(dataset.source_path, dataset.flame_mesh_dir, opt_hair, dataset.sh_degree)
    scene = Scene(dataset, gaussians, pointcloud_path=pointcloud_path_head, load_iteration=-1)
    gaussians.training_setup(opt)
    gaussians_hair.create_from_pcd(dataset.source_path, dataset.strand_scale)
    gaussians_hair.training_setup(opt, opt_hair)
    if checkpoint:
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if checkpoint_hair:
        model_params, first_iter = torch.load(checkpoint_hair)
        gaussians_hair.restore(model_params, opt_hair)
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

    with torch.no_grad():
        # Head gaussians data
        gaussians.mask_precomp = gaussians.get_label[..., 0] < 0.5
        gaussians.xyz_precomp = gaussians.get_xyz[gaussians.mask_precomp].detach()
        gaussians.opacity_precomp = gaussians.get_opacity[gaussians.mask_precomp].detach()
        gaussians.scaling_precomp = gaussians.get_scaling[gaussians.mask_precomp].detach()
        gaussians.rotation_precomp = gaussians.get_rotation[gaussians.mask_precomp].detach()
        gaussians.cov3D_precomp = gaussians.get_covariance(1.0)[gaussians.mask_precomp].detach()
        gaussians.shs_view = gaussians.get_features[gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1)**2)

    bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 100] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_hair(custom_cam, gaussians, gaussians_hair, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians_hair.initialize_gaussians_hair(iteration)
        gaussians_hair.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_hair(viewpoint_cam, gaussians, gaussians_hair, pipe, background)
    
        image = render_pkg["render"]
        mask = render_pkg["mask"]
        orient_angle = render_pkg["orient_angle"]
        orient_conf = render_pkg["orient_conf"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.original_mask.cuda()
        gt_orient_angle = viewpoint_cam.original_orient_angle.cuda()
        gt_orient_conf = viewpoint_cam.original_orient_conf.cuda()

        LCE = l1_loss(mask[:1], gt_mask[:1]) #, mask=(gt_mask[:1] < 0.5).detach())
        Ll1 = l1_loss(image, gt_image)
    
        # These losses are only applied for hair
        orient_weight = torch.ones_like(gt_mask[:1])
        if opt.use_gt_orient_conf: orient_weight = orient_weight * gt_orient_conf
        if not opt.train_orient_conf: orient_conf = None
        LOR = or_loss(orient_angle, gt_orient_angle, orient_conf, weight=orient_weight, mask=gt_mask[:1])

        # Diffusion loss
        LDF = gaussians_hair.LDiff if gaussians_hair.LDiff is not None else torch.zeros_like(LOR)

        if torch.isnan(Ll1).any(): Ll1 = torch.zeros_like(Ll1)
        if torch.isnan(LCE).any(): LCE = torch.zeros_like(Ll1)
        if torch.isnan(LOR).any(): LOR = torch.zeros_like(Ll1)
        if torch.isnan(LDF).any(): LDF = torch.zeros_like(Ll1)

        loss = (
            Ll1 * opt.lambda_dl1 + 
            LCE * opt.lambda_dmask +
            LOR * opt.lambda_dorient + 
            LDF * opt.lambda_dsds
        )
        loss.backward()

        iter_end.record()

        # Optimizer step
        if iteration < opt.iterations:
            for param in gaussians_hair.optimizer.param_groups[0]['params']:
                if param.grad is not None and param.grad.isnan().any():
                    gaussians_hair.optimizer.zero_grad()
                    print('NaN during backprop was found, skipping iteration...')
            gaussians_hair.optimizer.step()
            gaussians_hair.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, LCE, LOR, LDF, loss, l1_loss, ce_loss, or_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians_hair, render_hair, (pipe, background))

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(model_path_hair + "/checkpoints", exist_ok=True)
                torch.save((gaussians_hair.capture(), iteration), model_path_hair + "/checkpoints/" + str(iteration) + ".pth")
                # torch.save(gaussians_hair.strands_generator.texture_decoder.state_dict(), f'{model_path_hair}/checkpoints/texture_decoder.pth')

def prepare_output_and_logger(args, model_path_hair):    
    if not model_path_hair:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_path_hair = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(model_path_hair))
    os.makedirs(model_path_hair, exist_ok = True)
    with open(os.path.join(model_path_hair, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path_hair)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, LCE, LOR, LDF, loss, l1_loss, ce_loss, or_loss, elapsed, testing_iterations, scene : Scene, gaussians_hair, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ce_loss', LCE.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/or_loss', LOR.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/df_loss', LDF.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar("scaling/0", gaussians_hair.get_scaling[:, 0].mean().item(), iteration)
        tb_writer.add_scalar("scaling/1", gaussians_hair.get_scaling[:, 1].mean(), iteration)
        tb_writer.add_scalar("scaling/2", gaussians_hair.get_scaling[:, 2].mean(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        gaussians_hair.initialize_gaussians_hair(iteration)
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                ce_test = 0.0
                or_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, gaussians_hair, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    mask = torch.clamp(render_pkg["mask"], 0.0, 1.0)
                    orient_angle = torch.clamp(render_pkg["orient_angle"], 0.0, 1.0)
                    orient_conf = render_pkg["orient_conf"]
                    orient_conf_vis = (1 - 1 / (orient_conf + 1)) * mask[:1]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    gt_orient_angle = torch.clamp(viewpoint.original_orient_angle.to("cuda"), 0.0, 1.0)
                    gt_orient_conf = viewpoint.original_orient_conf.to("cuda")
                    gt_orient_conf_vis = (1 - 1 / (gt_orient_conf + 1)) * gt_mask[:1]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), F.pad(mask, (0, 0, 0, 0, 0, 3-mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient".format(viewpoint.image_name), vis_orient(orient_angle, mask[:1])[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient_conf".format(viewpoint.image_name), vis_orient(orient_angle, orient_conf_vis)[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), F.pad(gt_mask, (0, 0, 0, 0, 0, 3-gt_mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_mask[:1])[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient_conf".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_orient_conf_vis)[None], global_step=iteration)
                    l1_test += l1_loss(image * gt_mask[:1], gt_image * gt_mask[:1]).mean().double()
                    ce_test += ce_loss(mask[:1], gt_mask[:1]).mean().double()
                    or_test += or_loss(orient_angle, gt_orient_angle, mask=gt_mask[:1], weight=gt_orient_conf).mean().double()
                    psnr_test += psnr(image * gt_mask[:1], gt_image * gt_mask[:1]).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ce_test /= len(config['cameras'])
                or_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} CE {} OR {} PSNR {}".format(iteration, config['name'], l1_test, ce_test, or_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', ce_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - or_loss', or_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_checkpoint_hair", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--model_path_hair", type=str, default = None)
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path_hair)

    # Configuration of hair strands
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        opt_hair = yaml.load(replaced_conf, Loader=yaml.Loader)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), opt_hair, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.model_path_hair, args.pointcloud_path_head, args.start_checkpoint, args.start_checkpoint_hair, args.debug_from)

    # All done
    print("\nTraining complete.")