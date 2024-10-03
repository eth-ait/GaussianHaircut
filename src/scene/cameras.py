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
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.camera_opt_utils import lie
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 width, height,
                 image, mask_body, mask_hair, orient_angle, orient_conf, depth,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", white_background = False,
                 trainable_cameras = False, use_barf = False, trainable_intrinsics = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self._FoVx = FoVx + torch.zeros(1).cuda()
        self._FoVy = FoVy + torch.zeros(1).cuda()
        self.width = width
        self.height = height
        self.image_name = image_name
        self.trainable_cameras = trainable_cameras
        self.use_barf = use_barf
        self.trainable_intrinsics = trainable_intrinsics

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_mask_body = mask_body.clamp(0.0, 1.0).to(self.data_device)
        self.original_mask_hair = mask_hair.clamp(0.0, 1.0).to(self.data_device)
        self.original_mask = torch.cat([mask_hair, mask_body], dim=0)
        self.original_orient_angle = orient_angle.clamp(0.0, 1.0).to(self.data_device)
        self.original_orient_conf = orient_conf.to(self.data_device)

        # Renormalize the depth for visuals
        self.original_depth = depth.to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.original_image = self.original_image * self.original_mask_body + white_background * (1 - self.original_mask_body)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self._colmap_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self._world_view_transform = self._colmap_transform.transpose(0, 1)
        self._projection_matrix = getProjectionMatrix(
            znear=self.znear, 
            zfar=self.zfar, 
            fovX=self._FoVx, 
            fovY=self._FoVy
        ).transpose(0,1).cuda()
        self._full_proj_transform = (self._world_view_transform.unsqueeze(0).bmm(self._projection_matrix.unsqueeze(0))).squeeze(0)
        self._camera_center = self._world_view_transform.inverse()[3, :3]

        if self.trainable_cameras:
            if self.use_barf:
                self._rotation_res = nn.Parameter(torch.zeros(3).cuda())
                self._translation_res = nn.Parameter(torch.zeros(3).cuda())
            else:
                self._rotation_res = nn.Parameter(torch.eye(3, 3)[:2].view(-1).clone().cuda())
                self._translation_res = nn.Parameter(torch.zeros(3).cuda())

        if self.trainable_intrinsics:
            self._fov_res = nn.Parameter(torch.zeros(2).cuda())

    @property
    def FoVx(self):
        if self.trainable_intrinsics:
            return self._FoVx + self._fov_res[0]
        else:
            return self._FoVx

    @property
    def FoVy(self):
        if self.trainable_intrinsics:
            return self._FoVy + self._fov_res[1]
        else:
            return self._FoVy

    @property
    def _residual_transform(self):
        if self.use_barf:
            se3_refine = torch.cat([self._rotation_res, self._translation_res])
            pose_refine = lie.se3_to_SE3(se3_refine)
            residual_transform = torch.eye(4).cuda()
            residual_transform[:3] = pose_refine
        else:
            R_a = ortho2rotation(self._rotation_res)
            t_a = self._translation_res
            residual_transform = torch.eye(4).cuda()
            residual_transform[:3, :3] = R_a
            residual_transform[:3, 3] = t_a
        return residual_transform

    @property
    def world_view_transform(self):
        if self.trainable_cameras:
            world_view_transform = (self._colmap_transform @ self._residual_transform).transpose(0, 1)
            return world_view_transform
        else:
            return self._world_view_transform

    @property
    def projection_matrix(self):
        if self.trainable_intrinsics:
            self._projection_matrix = getProjectionMatrix(
                znear=self.znear, 
                zfar=self.zfar, 
                fovX=self.FoVx, 
                fovY=self.FoVy
            ).transpose(0,1)
        return self._projection_matrix

    @property
    def full_proj_transform(self):
        if self.trainable_cameras:
            return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        else:
            return self._full_proj_transform

    @property
    def camera_center(self):
        if self.trainable_cameras:
            return self.world_view_transform.inverse()[3, :3]
        else:
            return self._camera_center

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def ortho2rotation(poses):
    r"""
    poses: batch x 6
    From https://github.com/chrischoy/DeepGlobalRegistration/blob/master/core
    /registration.py#L16
    Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) 
    and Wei Dong (weidong@andrew.cmu.edu)
    """
    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(-1, keepdim=True)
        norm2 = (u ** 2).sum(-1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]

    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - proj_u2a(x, y_raw), dim=-1)
    z = torch.cross(x, y, dim=-1)

    return torch.stack([x, y, z], -1)