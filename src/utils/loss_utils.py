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
from torch.autograd import Variable
from math import exp, log, pi



def l1_loss(network_output, gt, weight = None, mask = None):
    loss = torch.abs(network_output - gt)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()

def ce_loss(network_output, gt):
    return F.binary_cross_entropy(network_output.clamp(1e-3, 1.0 - 1e-3), gt)

def or_loss(network_output, gt, confs = None, weight = None, mask = None):
    weight = torch.ones_like(gt[:1]) if weight is None else weight
    loss = torch.minimum(
        (network_output - gt).abs(),
        torch.minimum(
            (network_output - gt - 1).abs(), 
            (network_output - gt + 1).abs()
        ))
    loss = loss * pi
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss * weight

def dp_loss(pred, gt, pred_mask, gt_mask, eps=0.1):
    filter_fg = torch.logical_and(gt_mask >= 1 - eps, pred_mask >= 1 - eps).detach()
    
    if (filter_fg.sum() == 0).all():
        return None, pred, gt
    
    pred_fg = pred[filter_fg]
    gt_fg = gt[filter_fg]

    with torch.no_grad():    
        # # Subsample points
        # idx_1 = torch.argsort(gt_fg).detach()
        # idx_2 = torch.randperm(gt_fg.shape[0], device='cuda').detach()
        # to_penalize = torch.logical_or(
        #     torch.logical_and(idx_1 < idx_2, pred_fg[idx_1] > pred_fg[idx_2]),
        #     torch.logical_and(idx_1 > idx_2, pred_fg[idx_1] < pred_fg[idx_2])
        # ).detach()

        pred_q2, pred_q98 = torch.quantile(pred_fg, torch.tensor([0.02, 0.98]).cuda())
        gt_q2, gt_q98 = torch.quantile(gt_fg, torch.tensor([0.02, 0.98]).cuda())

    pred_aligned = ((pred - pred_q2.detach()) / (pred_q98.detach() - gt_q2.detach())).clamp(0, 1)
    gt_aligned = ((gt - gt_q2) / (gt_q98 - gt_q2)).clamp(0, 1)

    mask = gt_mask * pred_mask.detach()
    pred_masked = pred_aligned * mask + (1 - mask)
    gt_masked = gt_aligned * mask + (1 - mask)

    loss = (pred_masked - gt_masked).abs().mean()

    return loss, pred_masked, gt_masked

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average=True)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

