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
import math

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def vis_orient(orient_angle, mask):
    device = orient_angle.device
    deg = orient_angle * 180
    red = torch.clamp(1 - torch.abs(deg -  0.) / 45., 0, 1) + torch.clamp(1 - torch.abs(deg - 180.) / 45., 0, 1) # vertical
    green = torch.clamp(1 - torch.abs(deg - 90.) / 45., 0, 1) # horizontal
    magenta = torch.clamp(1 - torch.abs(deg - 45.) / 45., 0, 1) # diagonal down
    teal = torch.clamp(1 - torch.abs(deg - 135.) / 45., 0, 1) # diagonal up
    bgr = (
        torch.tensor([0, 0, 1])[:, None, None].to(device) * red +
        torch.tensor([0, 1, 0])[:, None, None].to(device) * green +
        torch.tensor([1, 0, 1])[:, None, None].to(device) * magenta +
        torch.tensor([1, 1, 0])[:, None, None].to(device) * teal
    )
    rgb = torch.stack([bgr[2], bgr[1], bgr[0]], dim=0)

    return rgb * mask

def vis_depth(depth):
    depth_vis = (depth + 1).log()
    depth_vis = (depth_vis - depth_vis.amin()) / (depth_vis.amax() / depth_vis.amin())
    return depth_vis