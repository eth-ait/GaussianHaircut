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
import numpy as np
from utils.general_utils import inverse_sigmoid
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation, parallel_transport
import math
import pickle as pkl
import sys
sys.path.append('../ext/NeuralHaircut/')
sys.path.append('../ext/NeuralHaircut/k-diffusion')
from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from src.hair_networks.strand_prior import Decoder



class GaussianModelHair:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_full_covariance=False):
            M = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = M.transpose(1, 2) @ M
            if return_full_covariance:
                return actual_covariance
            else:
                symm = strip_symmetric(actual_covariance)
                return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.label_activation = torch.sigmoid
        self.inverse_label_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.orient_conf_activation = torch.exp
        self.orient_conf_inverse_activation = torch.log

    def __init__(self, data_dir, flame_mesh_dir, strands_config, sh_degree):
        num_guiding_strands = strands_config['extra_args']['num_guiding_strands']
        self.num_guiding_strands = num_guiding_strands if num_guiding_strands is not None else 0
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._orient_conf = torch.empty(0)
        self._label = torch.empty(0)
        self.strands_generator = OptimizableTexturedStrands(
            **strands_config['textured_strands'], 
            diffusion_cfg=strands_config['diffusion_prior'],
            data_dir=data_dir,
            flame_mesh_dir=flame_mesh_dir,
            num_guiding_strands=num_guiding_strands
        ).cuda()
        self.color_decoder = Decoder(None, dim_hidden=128, num_layers=2, dim_out=3*(self.max_sh_degree+1)**2 + 1).cuda()
        self.optimizer = None
        self.scheduler = None
        self.setup_functions()

    def capture(self):
        return (
            self._scaling,
            self.active_sh_degree,
            self.strands_generator.state_dict(),
            self.color_decoder.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (
            self._scaling,
            self.active_sh_degree, 
            gen_dict,
            clr_dict,
            opt_dict,
            shd_dict,
        ) = model_args
        self.strands_generator.load_state_dict(gen_dict)
        self.color_decoder.load_state_dict(clr_dict)
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)
        self.scheduler.load_state_dict(shd_dict)

    @property
    def get_scaling(self):
        scaling = torch.ones_like(self.get_xyz)
        scaling[:, 0] = self._dir.norm(dim=-1) * 0.5
        scaling[:, 1:] = self.scale

        return scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_gdn(self):
        features_dc_gdn = self._features_dc_gdn
        features_rest_gdn = self._features_rest_gdn
        return torch.cat((features_dc_gdn, features_rest_gdn), dim=1)

    @property
    def get_opacity(self):
        return torch.ones_like(self.get_xyz[:, :1]) # self.opacity_activation(self._opacity)

    @property
    def get_label(self):
        return torch.ones_like(self.get_xyz[:, :1]) # self.label_activation(self._label)
    
    @property
    def get_orient_conf(self):
        return self.orient_conf_activation(self._orient_conf)

    @torch.no_grad()
    def filter_points(self, viewpoint_camera):
        # __forceinline__ __device__ bool in_frustum(int idx,
        #     const float* orig_points,
        #     const float* viewmatrix,
        #     const float* projmatrix,
        #     bool prefiltered,
        #     float3& p_view)
        # {
        #     float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

        #     // Bring points to screen space
        #     float4 p_hom = transformPoint4x4(p_orig, projmatrix);
        #     float p_w = 1.0f / (p_hom.w + 0.0000001f);
        #     float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
        #     p_view = transformPoint4x3(p_orig, viewmatrix);

        #     if (p_view.z <= 0.2f)
        #     {
        #         return false;
        #     }
        #     return true;
        # }        
        mean = self.get_xyz
        viewmatrix = viewpoint_camera.world_view_transform
        p_view = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        
        mask = p_view[:, [2]] > 0.2

        mask = torch.logical_and(mask, self.det != 0)

        # float mid = 0.5f * (cov.x + cov.z);
        # float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
        # float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
        # float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

        mid = 0.5 * (self.cov[:, [0]] + self.cov[:, [2]])
        sqrtD = (torch.clamp(mid**2 - self.det, min=0.1))**0.5
        lambda1 = mid + sqrtD
        lambda2 = mid - sqrtD
        my_radius = torch.ceil(3 * (torch.maximum(lambda1, lambda2))**0.5)

        # float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

        # __forceinline__ __device__ float ndc2Pix(float v, int S)
        # {
        #     return ((v + 1.0) * S - 1.0) * 0.5;
        # }
        
        point_image_x = ((self.p_proj[:, [0]] + 1) * viewpoint_camera.image_width - 1.0) * 0.5
        point_image_y = ((self.p_proj[:, [1]] + 1) * viewpoint_camera.image_height - 1.0) * 0.5

        # dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

        BLOCK_X = 16
        BLOCK_Y = 16

        grid_x = (viewpoint_camera.image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (viewpoint_camera.image_height + BLOCK_Y - 1) // BLOCK_Y

        # uint2 rect_min, rect_max;
        # getRect(point_image, my_radius, rect_min, rect_max, grid);

        # __forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
        # {
        #     rect_min = {
        #         min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        #         min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
        #     };
        #     rect_max = {
        #         min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        #         min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
        #     };
        # }
        
        rect_min_x = torch.clamp(((point_image_x - my_radius) / BLOCK_X).int(), min=0, max=grid_x)
        rect_min_y = torch.clamp(((point_image_y - my_radius) / BLOCK_Y).int(), min=0, max=grid_y)

        rect_max_x = torch.clamp(((point_image_x + my_radius + BLOCK_X - 1) / BLOCK_X).int(), min=0, max=grid_x)
        rect_max_y = torch.clamp(((point_image_y + my_radius + BLOCK_Y - 1) / BLOCK_Y).int(), min=0, max=grid_y)

        # if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        #     return;
        
        self.points_mask = torch.logical_and(mask, (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) != 0).squeeze()

        return self.points_mask

    def get_covariance(self, scaling_modifier = 1, return_full_covariance = False):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, return_full_covariance)

    def get_covariance_2d(self, viewpoint_camera, scaling_modifier = 1):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        J = torch.stack(
            [
                torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st column
                torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd column
                torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd column
            ],
            dim=-1 # stack columns into rows
        )

        W = viewmatrix[None, :3, :3]

        T = W @ J

        Vrk = self.get_covariance(scaling_modifier, return_full_covariance=True)

        cov = T.transpose(1, 2) @ Vrk.transpose(1, 2) @ T

        # J = torch.stack(
        #     [
        #         torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st row
        #         torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd row
        #         torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd row
        #     ],
        #     dim=-1 # stack rows into columns
        # )

        # W = viewmatrix[None, :3, :3]

        # T = J @ W

        # Vrk = self.get_covariance(viewpoint_camera, scaling_modifier, return_full_covariance=True)

        # cov = T @ Vrk @ T.transpose(1, 2)

        cov[:, 0, 0] += 0.3
        cov[:, 1, 1] += 0.3

        return torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]], dim=-1)

    def get_conic(self, viewpoint_camera, scaling_modifier = 1):
        self.cov = self.get_covariance_2d(viewpoint_camera, scaling_modifier)
        # mean = self.get_xyz

        # height = int(viewpoint_camera.image_height)
        # width = int(viewpoint_camera.image_width)
    
        # tan_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
        # tan_fovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # focal_y = height / (2.0 * tan_fovy)
        # focal_x = width / (2.0 * tan_fovx)

        # viewmatrix = viewpoint_camera.world_view_transform

        # t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        # tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        # limx = 1.3 * tan_fovx
        # limy = 1.3 * tan_fovy
        # txtz = tx / tz
        # tytz = ty / tz
        
        # tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        # ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        # zeros = torch.zeros_like(tz)

        # J = torch.stack(
        #     [
        #         torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st row
        #         torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd row
        #         torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd row
        #     ],
        #     dim=-2 # stack rows into columns
        # )

        # W = viewmatrix[None, :3, :3]

        # T = J @ W

        # Vrk = self.get_covariance(scaling_modifier, return_full_covariance=True)

        # cov = T @ Vrk @ T.transpose(1, 2)
        # cov[:, 0, 0] += 0.3
        # cov[:, 1, 1] += 0.3

        # cov = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]], dim=-1)

        # // Invert covariance (EWA algorithm)
        # float det = (cov.x * cov.z - cov.y * cov.y);
        # if (det == 0.0f)
        #     return;
        # float det_inv = 1.f / det;
        # float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
        self.det = self.cov[:, [0]] * self.cov[:, [2]] - self.cov[:, [1]]**2
        det_inv = 1. / (self.det + 0.0000001)
        conic = torch.stack([self.cov[:, 2], -self.cov[:, 1], self.cov[:, 0]], dim=-1) * det_inv
        # det = cov[:, [0]] * cov[:, [2]] - cov[:, [1]]**2
        # det_inv = (1. / (det + 1e-12)) * (det > 1e-12)
        # conic = cov * det_inv

        return conic

    def get_mean_2d(self, viewpoint_camera):
        # __forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
        # {
        #     float4 transformed = {
        #         matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        #         matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        #         matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        #         matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
        #     };
        #     return transformed;
        # }
        #
		# float4 p_hom = transformPoint4x4(p_orig, projmatrix);
		# float p_w = 1.0f / (p_hom.w + 0.0000001f);
		# p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
        projmatrix = viewpoint_camera.full_proj_transform
        p_hom = (self.get_xyz[:, None, :] @ projmatrix[None, :3, :] + projmatrix[None, [3]])[:, 0]
        p_w = 1.0 / (p_hom[:, [3]] + 0.0000001)
        self.p_proj = p_hom[:, :3] * p_w

        return self.p_proj

    def get_depths(self, viewpoint_camera):
        viewmatrix = viewpoint_camera.world_view_transform
        p_view = (self.get_xyz[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        return p_view[:, -1:]

    def get_direction_2d(self, viewpoint_camera):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        J = torch.stack(
            [
                torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st column
                torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd column
                torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd column
            ],
            dim=-1 # stack columns into rows
        )

        W = viewmatrix[None, :3, :3]

        T = W @ J

        dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (dir3D[:, None, :] @ T)[:, 0]

        return dir2D

    def generate_strands(self, iter, num_strands = -1):
        if num_strands != -1:
            p, _, _, _, _, z = self.strands_generator.forward_inference(num_strands)
            diffusion_dict = {}
        else:
            p, _, _, _, _, z, diffusion_dict = self.strands_generator(iter)
        self.num_strands = p.shape[0]
        self.strand_length = p.shape[1]

        self._xyz = (p[:, 1:] + p[:, :-1]).view(-1, 3) * 0.5
        self._dir = (p[:, 1:] - p[:, :-1]).view(-1, 3)
        # self._label = z[:, :1].view(self.num_strands, 1, 1).repeat(1, self.strand_length - 1, 1).view(-1, 1)
        z_app = z[:, 1:]
        
        if self.num_guiding_strands:
            self._xyz_gdn = self._xyz.view(self.num_strands, self.strand_length - 1 , 3)[:self.num_guiding_strands].view(-1, 3)
            self._dir_gdn = self._dir.view(self.num_strands, self.strand_length - 1 , 3)[:self.num_guiding_strands].view(-1, 3)
        else:
            self._xyz_gdn = self._xyz
            self._dir_gdn = self._dir

        # Assign spherical harmonics features
        if z_app.shape[1] == 3 * (self.max_sh_degree + 1) ** 2:
            features_dc, features_rest = z_app.view(self.num_strands, 1, 3 * (self.max_sh_degree + 1) ** 2).split([3, 3 * ((self.max_sh_degree + 1) ** 2 - 1)], dim=-1)
            features_dc = features_dc.repeat(1, self.strand_length - 1, 1)
            features_rest = features_rest.repeat(1, self.strand_length - 1, 1)        
        elif z_app.shape[1] == (self.strand_length - 1) * 3 * (self.max_sh_degree + 1) ** 2:
            features_dc, features_rest = z_app.view(self.num_strands, self.strand_length - 1, 3 * (self.max_sh_degree + 1) ** 2).split([3, 3 * ((self.max_sh_degree + 1) ** 2 - 1)], dim=-1)
        elif z_app.shape[1] == 64:
            features_dc, features_rest, orient_conf = self.color_decoder(z_app).split([3, 3 * ((self.max_sh_degree + 1) ** 2 - 1), 1], dim=-1)

        self._features_dc = features_dc.reshape(self.num_strands * (self.strand_length - 1), 1, 3)
        self._features_rest = features_rest.reshape(self.num_strands * (self.strand_length - 1), (self.max_sh_degree + 1) ** 2 - 1, 3)
        self._orient_conf = orient_conf.reshape(self.num_strands * (self.strand_length - 1), 1)

        if self.num_guiding_strands:
            self._features_dc_gdn = features_dc[:self.num_guiding_strands].reshape(-1, 1, 3)
            self._features_rest_gdn = features_rest[:self.num_guiding_strands].reshape(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
        else:
            self._features_dc_gdn = self._features_dc
            self._features_rest_gdn = self._features_rest

        return diffusion_dict

    def initialize_gaussians_hair(self, iter, num_strands=-1):
        diffusion_dict = self.generate_strands(iter, num_strands)
        
        # Assign geometric features        
        self._rotation = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(self._xyz[:, :1]),
                    torch.zeros_like(self._xyz[:, :2])
                ],
                dim=-1
            ),
            b=self._dir
        ).view(-1, 4) # rotation parameters that align x-axis with the segment direction

        if 'L_diff' in diffusion_dict.keys():
            self.LDiff = diffusion_dict['L_diff']
        else:
            self.LDiff = None

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, data_path, scale = 1e-3):
        with torch.no_grad():
            self.generate_strands(0)

        scene_transform = pkl.load(open(os.path.join(data_path, "scale.pickle"), 'rb'))
        self.scale = scale * scene_transform['scale'] * torch.ones(1, device="cuda")

    def training_setup(self, training_args, training_args_hair):
        self.optimizer = torch.optim.AdamW(list(self.strands_generator.parameters()) + list(self.color_decoder.parameters()), training_args_hair['general']['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=training_args.iterations, eta_min=1e-4)

    def update_learning_rate(self, iteration):
        self.scheduler.step()