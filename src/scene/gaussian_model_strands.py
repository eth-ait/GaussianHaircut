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
from utils.general_utils import strip_symmetric, get_expon_lr_func, build_scaling_rotation, parallel_transport
import math
import pickle as pkl
import sys
import trimesh
from pysdf import SDF
sys.path.append('../ext/NeuralHaircut/')
sys.path.append('../ext/NeuralHaircut/k-diffusion')
from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from src.hair_networks.strand_prior import Decoder, Encoder



class GaussianModelCurves:
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
        self.flame_mesh_dir = flame_mesh_dir
        self.color_decoder = Decoder(None, dim_hidden=128, num_layers=2, dim_out=3*(self.max_sh_degree+1)**2 + 1).cuda()
        self.strands_encoder = Encoder(None).eval().cuda()
        self.strands_encoder.load_state_dict(torch.load(f'../ext/NeuralHaircut/pretrained_models/strand_prior/strand_ckpt.pth')['encoder'])
        self.optimizer = None
        self.use_sds = True
        self.setup_functions()

    def capture(self):
        return (
            self._pts,
            self._features_dc,
            self._features_rest,
            self.active_sh_degree,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (
            self._pts,
            self._features_dc,
            self._features_rest,
            self.active_sh_degree, 
            opt_dict,
        ) = model_args
        self.pts_origins = self._pts[:, :1]
        self._dirs = self._pts[:, 1:] - self._pts[:, :-1]
        self._orient_conf = torch.ones_like(self._features_dc[:, :1, 0])
        try:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
        except:
            print('Failed to load optimizer')

    @property
    def get_scaling(self):
        return self._scaling
    
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
    def get_opacity(self):
        return torch.ones_like(self.get_xyz[:, :1])

    @property
    def get_label(self):
        return torch.ones_like(self.get_xyz[:, :1])
    
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

    def initialize_gaussians_hair(self):
        self._pts = self.pts_origins + torch.cat([torch.zeros_like(self.pts_origins), torch.cumsum(self._dirs, dim=1)], dim=1)
        self._dir = self._dirs.view(-1, 3)
        self._xyz = (self._pts[:, 1:] + self._pts[:, :-1]).view(-1, 3) * 0.5
        
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

        self._scaling = torch.ones_like(self.get_xyz)
        self._scaling[:, 0] = self._dir.norm(dim=-1) * 0.5
        self._scaling[:, 1:] = self.scale

        if self.use_sds:
            # Encode the guiding strands into the latent vectors
            NUM_GUIDING_STRANDS = 1000
            idx = torch.randint(low=0, high=self.num_strands, size=(NUM_GUIDING_STRANDS,), device="cuda")
            uvs_gdn = self.uvs[idx]
            pts_gdn_local = (torch.inverse(self.local2world[idx][:, None]) @ (self._pts[idx] - self.pts_origins[idx])[..., None])[..., 0]
            v_gdn_local = (pts_gdn_local[:, 1:] - pts_gdn_local[:, :-1]) * self.strands_generator.scale_decoder
            z_gdn = self.strands_encoder(pts_gdn_local  * self.strands_generator.scale_decoder)[:, :64]
            # print((self.z_geom[idx] - z_gdn).abs().sum(-1).mean())
            # rec_v = self.strands_generator.strand_decoder(z_gdn) / self.strands_generator.scale_decoder
            # rec_pts_gdn_local = torch.cat(
            #     [
            #         torch.zeros_like(rec_v[:, -1:, :]), 
            #         torch.cumsum(rec_v, dim=1)
            #     ], 
            #     dim=1
            # )
            # print(((rec_pts_gdn_local - self.p_local[idx]).abs().sum(-1) / (self.p_local[idx].abs().sum(-1) + 1e-7)).mean())
            # raise
            grid = torch.linspace(start=-1, end=1, steps=self.strands_generator.diffusion_input + 1, device="cuda")
            grid = (grid[1:] + grid[:-1]) / 2
            uvs_sds = torch.stack(torch.meshgrid(grid, grid, indexing='xy'), dim=-1).view(-1, 2)

            # Find K nearest neighbours for each of the interpolated points in the UV space
            K = 4

            dist = ((uvs_sds.view(-1, 1, 2) - uvs_gdn.view(1, -1, 2))**2).sum(-1) # num_sds_strands x num_guiding_strands
            knn_dist, knn_idx = torch.sort(dist, dim=1)
            w = 1 / (knn_dist[:, :K] + 1e-7)
            w = w / w.sum(dim=-1, keepdim=True)
            
            z_sds_nearest = z_gdn[knn_idx[:, 0]]
            z_sds_bilinear = (z_gdn[knn_idx[:, :K]] * w[:, :, None]).sum(dim=1)
            
            # Calculate cosine similarity between neighbouring guiding strands to get blending alphas (eq. 4 of HAAR)
            knn_v = v_gdn_local[knn_idx[:, :K]]
            csim_full = torch.nn.functional.cosine_similarity(knn_v.view(-1, K, 1, 99, 3), knn_v.view(-1, 1, K, 99, 3), dim=-1).mean(-1) # num_guiding_strands x K x K
            j, k = torch.triu_indices(K, K, device=csim_full.device).split([1, 1], dim=0)
            i = torch.arange(NUM_GUIDING_STRANDS, device=csim_full.device).repeat_interleave(j.shape[1])
            j = j[0].repeat(NUM_GUIDING_STRANDS)
            k = k[0].repeat(NUM_GUIDING_STRANDS)
            csim = csim_full[i, j, k].view(NUM_GUIDING_STRANDS, -1).mean(-1)
            
            alpha = torch.where(csim <= 0.9, 1 - 1.63 * csim**5, 0.4 - 0.4 * csim)
            alpha_sds = (alpha[knn_idx[:, :K]] * w).sum(dim=1)[:, None]
            z_sds = z_sds_nearest * alpha_sds + z_sds_bilinear * (1 - alpha_sds)
            
            diffusion_texture = z_sds.view(1, self.strands_generator.diffusion_input, self.strands_generator.diffusion_input, 64).permute(0, 3, 1, 2)

            noise = torch.randn_like(diffusion_texture)
            sigma = self.strands_generator.sample_density([diffusion_texture.shape[0]], device='cuda')
            mask = None
            if self.strands_generator.diffuse_mask is not None:
                mask = torch.nn.functional.interpolate(
                    self.strands_generator.diffuse_mask[None][None], 
                    size=(self.strands_generator.diffusion_input, self.strands_generator.diffusion_input)
                )
            L_diff, pred_image, noised_image = self.strands_generator.model_ema.loss_wo_logvar(diffusion_texture, noise, sigma, mask=mask, unet_cond=None)

            self.Lsds = L_diff.mean()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, data_path, model_params, num_strands, spatial_lr_scale):
        with torch.no_grad():
            (
                _scaling,
                active_sh_degree, 
                gen_dict,
                clr_dict,
                opt_dict,
                shd_dict,
            ) = model_params
            self.strands_generator.load_state_dict(gen_dict)
            self.color_decoder.load_state_dict(clr_dict)

            self.spatial_lr_scale = spatial_lr_scale
            print(self.spatial_lr_scale)

            pts, uvs, local2world, p_local, z_geom, z = self.strands_generator.forward_inference(num_strands)
            self.pts_origins = pts[:, :1]
            self.uvs = uvs
            self.local2world = local2world
            self.z_geom = z_geom
            self.p_local = p_local
            dirs = pts[:, 1:] - pts[:, :-1]
            self.num_strands = pts.shape[0]
            self.strand_length = pts.shape[1]
            
            label = z[:, 0]
            z_app = z[:, 1:]

            features_dc, features_rest, orient_conf = self.color_decoder(z_app).split([3, 3 * ((self.max_sh_degree + 1) ** 2 - 1), 1], dim=-1)

        # # Prune hair strands with low label and the ones that intersect the FLAME mesh
        # mesh = trimesh.load(f'{self.flame_mesh_dir}/stage_3/mesh_final.obj')
        # sdf_handle = SDF(mesh.vertices, mesh.faces)
        # L = pts.shape[1]

        # p_npy = pts.detach().cpu().numpy()
        # sdf = sdf_handle(p_npy.reshape(-1, 3))
        # mask = (sdf.reshape(-1, L) < 0).mean(axis=1) >= 0.5
        mask = torch.ones_like(pts[..., 0, 0]).bool()
        print(f'Pruning {sum(~mask)} strands that intersect the head mesh')

        # mask = torch.logical_and(label >= 0.5, torch.from_numpy(mask).cuda())
        self.num_strands = sum(mask)

        self.pts_origins = self.pts_origins[mask]
        self.uvs = self.uvs[mask]
        self.local2world = self.local2world[mask]
        self.z_geom = z_geom[mask]
        self.p_local = p_local[mask]
        self._dirs = nn.Parameter(dirs[mask].contiguous().clone().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc[mask].reshape(-1, 1, 3).contiguous().clone().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest[mask].reshape(-1, (self.max_sh_degree + 1) ** 2 - 1, 3).contiguous().clone().requires_grad_(True))
        self._orient_conf = nn.Parameter(orient_conf[mask][:, None, :].reshape(-1, 1).contiguous().clone().requires_grad_(True))
        scene_transform = pkl.load(open(os.path.join(data_path, "scale.pickle"), 'rb'))
        self.scale = 1e-3 * scene_transform['scale'] * torch.ones(1, device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._dirs], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._orient_conf], 'lr': training_args.orient_conf_lr, "name": "orient_conf"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr