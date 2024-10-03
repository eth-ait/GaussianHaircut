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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_rotation
import math



class GaussianModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.label_activation = torch.sigmoid
        self.inverse_label_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.orient_conf_activation = torch.exp
        self.orient_conf_inverse_activation = torch.log

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._covariance = torch.empty(0)
        self._opacity = torch.empty(0)
        self._orient_conf = torch.empty(0)
        self._label = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._covariance,
            self._opacity,
            self._orient_conf,
            self._label,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args = None):
        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._orient_conf,
            self._label,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
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
        return self.opacity_activation(self._opacity)
    
    @property
    def get_label(self):
        return self.label_activation(self._label)
    
    @property
    def get_orient_conf(self):
        # scaling = self.get_scaling
        # i = torch.arange(scaling.shape[0], device='cuda')[:, None].repeat(1, 3).view(-1)
        # j = scaling.argsort(dim=-1, descending=True).view(-1)
        # sorted_S = scaling[i, j].view(-1, 3)
        # gaussian_shape_conf = (1 - sorted_S[:, [1]] / sorted_S[:, [0]]) * (1 - sorted_S[:, [2]] / sorted_S[:, [0]])

        return self.orient_conf_activation(self._orient_conf) # * gaussian_shape_conf

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
        viewmatrix = viewpoint_camera.world_view_transform

        xyz_view = (self.get_xyz[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        mask = xyz_view[:, [2]] > 0.2

        det = self.cov2d[:, [0]] * self.cov2d[:, [2]] - self.cov2d[:, [1]]**2
        mask = torch.logical_and(mask, det != 0)

        # float mid = 0.5f * (cov.x + cov.z);
        # float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
        # float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
        # float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

        mid = 0.5 * (self.cov2d[:, [0]] + self.cov2d[:, [2]])
        sqrtD = (torch.clamp(mid**2 - det, min=0.1))**0.5
        lambda1 = mid + sqrtD
        lambda2 = mid - sqrtD
        my_radius = torch.ceil(3 * (torch.maximum(lambda1, lambda2))**0.5)

        # float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

        # __forceinline__ __device__ float ndc2Pix(float v, int S)
        # {
        #     return ((v + 1.0) * S - 1.0) * 0.5;
        # }
        
        point_image_x = ((self.xyz_proj[:, [0]] + 1) * viewpoint_camera.image_width - 1.0) * 0.5
        point_image_y = ((self.xyz_proj[:, [1]] + 1) * viewpoint_camera.image_height - 1.0) * 0.5

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
        self.scaling = self.get_scaling
        s = self.scaling * scaling_modifier
        r = self._rotation

        S = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        self.R = build_rotation(r)

        S[:,0,0] = s[:,0]
        S[:,1,1] = s[:,1]
        S[:,2,2] = s[:,2]

        M = S @ self.R

        self.cov_full = M.transpose(1, 2) @ M
        self.cov = strip_symmetric(self.cov_full)

        if return_full_covariance:
            return self.cov_full
        else:
            return self.cov

    def get_covariance_2d(self, viewpoint_camera, scaling_modifier = 1):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = torch.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = torch.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        xyz_view = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = xyz_view[:, 0], xyz_view[:, 1], xyz_view[:, 2]

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

        self.proj_transform_cov = W @ J

        cov_full = self.get_covariance(scaling_modifier, return_full_covariance=True)

        self.cov2d_full = self.proj_transform_cov.transpose(1, 2) @ cov_full.transpose(1, 2) @ self.proj_transform_cov

        self.cov2d_full[:, 0, 0] += 0.3
        self.cov2d_full[:, 1, 1] += 0.3

        self.cov2d = torch.stack([self.cov2d_full[:, 0, 0], self.cov2d_full[:, 0, 1], self.cov2d_full[:, 1, 1]], dim=-1)

        return self.cov2d

    def get_conic(self, viewpoint_camera, scaling_modifier = 1):
        self.cov2d = self.get_covariance_2d(viewpoint_camera, scaling_modifier)
        # // Invert covariance (EWA algorithm)
        # float det = (cov.x * cov.z - cov.y * cov.y);
        # if (det == 0.0f)
        #     return;
        # float det_inv = 1.f / det;
        # float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
        det = self.cov2d[:, [0]] * self.cov2d[:, [2]] - self.cov2d[:, [1]]**2
        det_inv = 1. / (det + 1e-12)
        self.conic = torch.stack([self.cov2d[:, 2], -self.cov2d[:, 1], self.cov2d[:, 0]], dim=-1) * det_inv

        return self.conic

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
        self.xyz_proj = p_hom[:, :3] * p_w

        return self.xyz_proj

    def get_depths(self, viewpoint_camera):
        viewmatrix = viewpoint_camera.world_view_transform
        xyz_view = (self.get_xyz[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        return xyz_view[:, -1:]

    def get_direction_2d(self, viewpoint_camera):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = torch.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = torch.tan(viewpoint_camera.FoVy * 0.5)

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

        i = torch.arange(self.scaling.shape[0], device='cuda')[:, None].repeat(1, 3).view(-1)
        j = self.scaling.argsort(dim=-1, descending=True).view(-1)
        sorted_R = self.R[i, j].view(-1, 3, 3)
        sorted_S = self.scaling[i, j].view(-1, 3)
        self._dir = sorted_R[:, 0] * sorted_S[:, 0, None]

        # dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (self._dir[:, None, :] @ T)[:, 0]

        return dir2D

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._label = nn.Parameter(inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(True))
        self._orient_conf = nn.Parameter(torch.zeros_like(self._label).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._label], 'lr': training_args.label_lr, "name": "label"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        
        if training_args.train_orient_conf:
            l.append({'params': [self._orient_conf], 'lr': training_args.orient_conf_lr, "name": "orient_conf"})

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

    def construct_list_of_attributes(self, remove_label=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('orient_conf')
        if not remove_label:
            l.append('label_0')
        if len(self._covariance):
            for i in range(self._covariance.shape[1]):
                l.append('cov_{}'.format(i))
        else:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        mkdir_p(dir)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        orients_conf = self._orient_conf.detach().cpu().numpy()
        labels = self._label.detach().cpu().numpy()
        if len(self._covariance):
            covariance = self._covariance.detach().cpu().numpy()
        else:
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if len(self._covariance):
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, orients_conf, labels, covariance), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, orients_conf, labels, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(f'{dir}/raw_{name}')

        if not len(self._covariance):
            # A hack to not re-write the visualization software
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(remove_label=True)]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, orients_conf, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        try:
            orients_conf = np.asarray(plydata.elements[0]["orient_conf"])[..., np.newaxis]
        except:
            orients_conf = np.zeros((opacities.shape[0], 1))

        labels = np.zeros((xyz.shape[0], 1))
        labels[:, 0] = np.asarray(plydata.elements[0]["label_0"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        cov_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cov_")]
        cov_names = sorted(cov_names, key = lambda x: int(x.split('_')[-1]))
        covs = np.zeros((xyz.shape[0], len(cov_names)))
        for idx, attr_name in enumerate(cov_names):
            covs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._orient_conf = nn.Parameter(torch.tensor(orients_conf, dtype=torch.float, device="cuda").requires_grad_(True))
        self._label = nn.Parameter(torch.tensor(labels, dtype=torch.float, device="cuda").requires_grad_(True))
        if len(scale_names): self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        if len(rot_names): self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if len(cov_names): self._covariance = nn.Parameter(torch.tensor(covs, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._label = optimizable_tensors["label"]
        self._orient_conf = optimizable_tensors["orient_conf"] if "orient_conf" in optimizable_tensors.keys() else torch.zeros_like(self._xyz[:, :1])
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        if len(self.max_radii2D):
            self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_orient_confs, new_labels, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "orient_conf": new_orient_confs,
             "label": new_labels,
             "scaling" : new_scaling,
             "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._label = optimizable_tensors["label"]
        self._orient_conf = optimizable_tensors["orient_conf"] if "orient_conf" in optimizable_tensors.keys() else torch.zeros_like(self._label)
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_orient_conf = self._orient_conf[selected_pts_mask].repeat(N,1)
        new_label = self._label[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_orient_conf, new_label, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_orient_confs = self._orient_conf[selected_pts_mask]
        new_labels = self._label[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_orient_confs, new_labels, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = self.get_opacity.squeeze() < min_opacity
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1