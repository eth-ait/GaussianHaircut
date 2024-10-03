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
from utils.loss_utils import l1_loss, ssim, or_loss
from utils.general_utils import get_expon_lr_func
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, vis_orient
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle as pkl
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if dataset.trainable_cameras or dataset.trainable_intrinsics:
        params_cam_rotation = {}
        params_cam_translation = {}
        params_cam_fov = {}
        for k in scene.train_cameras.keys():
            for camera in scene.train_cameras[k]:
                if dataset.trainable_cameras:
                    params_cam_rotation[camera.image_name] = camera._rotation_res
                    params_cam_translation[camera.image_name] = camera._translation_res
                if dataset.trainable_intrinsics:
                    params_cam_fov[camera.image_name] = camera._fov_res
        params_cam = list(params_cam_rotation.values()) + list(params_cam_translation.values()) + list(params_cam_fov.values())
        l = [
            {'params': list(params_cam_rotation.values()), 'lr': opt.cam_rotation_lr, "name": "rotation"},
            {'params': list(params_cam_translation.values()), 'lr': opt.cam_translation_lr_init * gaussians.spatial_lr_scale, "name": "translation"},
            {'params': list(params_cam_fov.values()), 'lr': opt.cam_fov_lr, "name": "fov"}
        ]

        optimizer_cameras = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        translation_scheduler_args = get_expon_lr_func(lr_init=opt.cam_translation_lr_init * gaussians.spatial_lr_scale,
                                                       lr_final=opt.cam_translation_lr_final * gaussians.spatial_lr_scale,
                                                       max_steps=opt.cam_lr_max_steps)

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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    
        image = render_pkg["render"]
        mask = render_pkg["mask"]
        orient_angle = render_pkg["orient_angle"]
        orient_conf = render_pkg["orient_conf"]
        viewspace_point_tensor = render_pkg["viewspace_points"] 
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.original_mask.cuda()
        gt_orient_angle = viewpoint_cam.original_orient_angle.cuda()
        gt_orient_conf = viewpoint_cam.original_orient_conf.cuda()

        Ll1 = l1_loss(image, gt_image, mask=gt_mask[1:].detach())
        Lssim = (1.0 - ssim(image * gt_mask[1:], gt_image * gt_mask[1:]))
        Lmask = l1_loss(mask, gt_mask)

        orient_weight = torch.ones_like(gt_mask[:1]) * gt_orient_conf
        Lorient = or_loss(orient_angle, gt_orient_angle, orient_conf, weight=orient_weight, mask=gt_mask[:1])
        
        if torch.isnan(Lorient).any(): Lorient = torch.zeros_like(Ll1)

        loss = (
            Ll1 * opt.lambda_dl1 + 
            Lssim * opt.lambda_dssim + 
            Lmask * opt.lambda_dmask + 
            Lorient * opt.lambda_dorient
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Lmask, Lorient, loss, l1_loss, or_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._label, gaussians._scaling, gaussians._rotation]:
                    if param.grad is not None and param.grad.isnan().any():
                        gaussians.optimizer.zero_grad(set_to_none = True)
                        print('NaN during backprop was found, skipping iteration...')

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration < opt.iterations_cam and dataset.trainable_cameras:
                ''' Learning rate scheduling per step '''
                for param_group in optimizer_cameras.param_groups:
                    if param_group["name"] == "translation":
                        lr = translation_scheduler_args(iteration)
                        param_group['lr'] = lr
                
                for param in params_cam:
                    if param.grad is not None and param.grad.isnan().any():
                        optimizer_cameras.zero_grad(set_to_none = True)
                        print('NaN during backprop was found, skipping iteration...')

                optimizer_cameras.step()
                optimizer_cameras.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + "/checkpoints", exist_ok=True)
                os.makedirs(scene.model_path + "/cameras", exist_ok=True)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/checkpoints/" + str(iteration) + ".pth")
                if dataset.trainable_cameras:
                    pkl.dump((params_cam_rotation, params_cam_translation, params_cam_fov), open(scene.model_path + "/cameras/" + str(iteration) + ".pkl", 'wb'))
                projection_all = {}
                for camera in scene.train_cameras[1.0]:
                    projection_all[camera.image_name] = camera.full_proj_transform.cpu()
                pkl.dump(projection_all, open(scene.model_path + "/cameras/" + str(iteration) + "_matrices.pkl", 'wb'))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Lmask, Lorient, loss, l1_loss, or_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ce_loss', Lmask.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/or_loss', Lorient.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                Ll1_test = 0.0
                Lmask_test = 0.0
                Lorient_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
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
                    Ll1_test += l1_loss(image, gt_image).mean().double()
                    Lmask_test += l1_loss(mask, gt_mask).mean().double()
                    Lorient_test += or_loss(orient_angle, gt_orient_angle, mask=gt_mask[:1], weight=gt_orient_conf).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                Ll1_test /= len(config['cameras'])
                Lmask_test /= len(config['cameras'])
                Lorient_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} CE {} OR {} PSNR {}".format(iteration, config['name'], Ll1_test, Lmask_test, Lorient_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', Ll1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', Lmask_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - or_loss', Lorient_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/label_histogram", scene.gaussians.get_label, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 5_000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
