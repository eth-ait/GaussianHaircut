from PIL import Image
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage.filters import difference_of_gaussians
import math
import os
import tqdm
import cv2
import argparse
from torchvision.transforms import Resize, InterpolationMode
import torch
from torch import nn
from torch.nn import functional as F



def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def generate_gabor_filters(num_frequencies, num_filters, num_sigmas_x, num_sigmas_y, num_offsets):
    thetas = np.linspace(0, math.pi * (num_filters - 1) / num_filters, num_filters)
    offsets = np.linspace(0, math.pi * (num_offsets - 1) / num_offsets, num_offsets)
    sigmas_x = [1.8] if num_sigmas_x == 1 else 2**np.arange(num_sigmas_x)
    sigmas_y = [2.4] if num_sigmas_y == 1 else 2**np.arange(num_sigmas_y)
    frequencies = [0.23] if num_frequencies == 1 else 2.0**(-np.arange(num_frequencies))
    weight_list = []
    kernel_size = 0
    for theta in thetas:
        for sigma_x in sigmas_x:
            for sigma_y in sigmas_y:
                for offset in offsets:
                    for frequency in frequencies:
                        weight = np.real(gabor_kernel(frequency=frequency, theta=math.pi - theta, sigma_x=sigma_x, sigma_y=sigma_y, offset=offset))
                        kernel_size = max(max(weight.shape[0], weight.shape[1]), kernel_size)
                        weight_list.append(weight)
    kernel_size += 1 - (kernel_size % 2)
    weight = np.zeros((num_filters * num_sigmas_x * num_sigmas_y * num_offsets * num_frequencies, kernel_size, kernel_size))
    for i in range(weight.shape[0]):
        kernel_size_i = weight_list[i].shape[:2]
        pad_y = (kernel_size - kernel_size_i[0]) // 2
        pad_x = (kernel_size - kernel_size_i[1]) // 2
        weight[i, pad_y : pad_y + kernel_size_i[0], pad_x : pad_x + kernel_size_i[1]] = weight_list[i]
    kernel = nn.Conv2d(1, len(thetas) * len(sigmas_x) * len(sigmas_y) * len(offsets), kernel_size=kernel_size, bias=False)
    kernel.weight.data = torch.from_numpy(weight[:, None, :, :]).float().cuda()

    return kernel, thetas


def calc_orients(img, dog_low, dog_high, num_frequencies, num_filters, num_sigmas_x, num_sigmas_y, num_offsets, patch_size):
    H, W = img.shape[:2]
    gray_img = rgb2gray(img)
    filtered_image = difference_of_gaussians(gray_img, dog_low, dog_high)
    filtered_image_cuda = torch.from_numpy(filtered_image).float().cuda()

    kernel, thetas = generate_gabor_filters(num_frequencies, num_filters, num_sigmas_x, num_sigmas_y, num_offsets)
    assert kernel.kernel_size[0] == kernel.kernel_size[1]
    padding = kernel.kernel_size[0] // 2
    filtered_image_cuda_padded = F.pad(filtered_image_cuda, (padding, padding, padding, padding))
    thetas_cuda = torch.from_numpy(thetas).float().cuda()

    orients_deg = torch.zeros(H, W).long()
    orients_var = torch.zeros(H, W)
    
    i_grid, j_grid = torch.meshgrid(*[torch.arange(patch_size)]*2)

    with torch.no_grad():
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                input_patch = filtered_image_cuda_padded[i : i + patch_size + 2 * padding, j : j + patch_size + 2 * padding]
                input_patch_height, input_patch_width = input_patch.shape
                output_patch_height = input_patch_height - 2 * padding
                output_patch_width = input_patch_width - 2 * padding
                gabor_filtered_patch = kernel(input_patch[None, None])
                F_patch = gabor_filtered_patch.abs().view(num_filters, num_sigmas_x * num_sigmas_y * num_offsets * num_frequencies, output_patch_height, output_patch_width)
                F_patch_norm = F.normalize(F_patch, p=1.0, dim=0)
                orients_deg_patch = F_patch.argmax(0) # [num_sigmas_x * num_sigmas_y * num_offsets * num_frequencies, output_patch_height, output_patch_width]
                orients_rad_patch = orients_deg_patch / num_filters * math.pi
                dists = torch.minimum((orients_rad_patch[None] - thetas_cuda[:, None, None, None]).abs(),
                                    torch.minimum((orients_rad_patch[None] - thetas_cuda[:, None, None, None] - math.pi).abs(),
                                                  (orients_rad_patch[None] - thetas_cuda[:, None, None, None] + math.pi).abs()))
                orients_var_patch = (dists**2 * F_patch_norm).sum(0)
                
                # Select the filter output with the minimum variance
                idx_patch = orients_var_patch.argmin(0).reshape(-1)
                i_grid_patch = i_grid[:output_patch_height, :output_patch_width].reshape(-1)
                j_grid_patch = j_grid[:output_patch_height, :output_patch_width].reshape(-1)
                orients_deg[i : i + patch_size, j : j + patch_size] = orients_deg_patch[idx_patch, i_grid_patch, j_grid_patch].view(output_patch_height, output_patch_width).cpu()
                orients_var[i : i + patch_size, j : j + patch_size] = orients_var_patch[idx_patch, i_grid_patch, j_grid_patch].view(output_patch_height, output_patch_width).cpu()
    
    orients_deg = orients_deg.numpy()
    orients_var = orients_var.numpy()

    return orients_deg, orients_var, filtered_image


def main(args):

    os.makedirs(args.orient_dir, exist_ok=True)
    os.makedirs(args.conf_dir, exist_ok=True)
    os.makedirs(args.filtered_img_dir, exist_ok=True)
    os.makedirs(args.vis_img_dir, exist_ok=True)

    img_list = sorted(os.listdir(args.mask_path))

    for img_name in tqdm.tqdm(img_list):
        basename = img_name.split('.')[0]
        img = Image.open(os.path.join(args.img_path, img_name))
        img = np.array(img)
        
        orientation_map_, confidence_map_, filtered_img_ = calc_orients(img, args.dog_low, args.dog_high, args.num_frequencies, args.num_filters, args.num_sigmas_x, args.num_sigmas_y, args.num_offsets, args.patch_size)
        filtered_img_ = (filtered_img_ - filtered_img_.min()) / (filtered_img_.max() - filtered_img_.min()) * 255

        orientation_map = orientation_map_.astype('uint8')
        confidence_map = confidence_map_
        filtered_img = filtered_img_

        # if args.crop_size != -1:
        #     orientation_map_ = np.asarray(Image.fromarray(orientation_map_.astype('uint8')).resize(crop_size, Image.BICUBIC))
        #     confidence_map_ = F.interpolate(torch.from_numpy(confidence_map_)[None, None], size=crop_size[::-1], mode='bilinear').numpy()[0, 0]
        #     filtered_img_ = np.asarray(Image.fromarray(filtered_img_.astype('uint8')).resize(crop_size, Image.BICUBIC))

        # # Uncrop
        # orientation_map = np.zeros_like(mask).astype('float')
        # orientation_map[u:d, l:r] = orientation_map_
        # confidence_map = np.zeros_like(mask).astype('float')
        # confidence_map[u:d, l:r] = confidence_map_
        # filtered_img = np.zeros_like(mask).astype('float')
        # filtered_img[u:d, l:r] = filtered_img_
        
        rad = orientation_map
        mask = np.asarray(Image.open(os.path.join(args.mask_path, img_name))) / 255.
        red = np.clip(1 - np.abs(rad -  0.) / 45., a_min=0, a_max=1) + np.clip(1 - np.abs(rad - 180.) / 45., a_min=0, a_max=1)
        green = np.clip(1 - np.abs(rad - 90.) / 45., a_min=0, a_max=1)
        magenta = np.clip(1 - np.abs(rad - 45.) / 45., a_min=0, a_max=1)
        teal = np.clip(1 - np.abs(rad - 135.) / 45., a_min=0, a_max=1)
        rgb = (
            np.array([0, 0, 1])[None, None] * red[..., None] +
            np.array([0, 1, 0])[None, None] * green[..., None] +
            np.array([1, 0, 1])[None, None] * magenta[..., None] +
            np.array([1, 1, 0])[None, None] * teal[..., None]
        )
        # norm = (r + g + b) * 0.5
        # b = np.zeros_like(r)
        norm = np.ones_like(rgb[..., 0])

        vis_img = np.clip(rgb / norm[..., None], a_min=0, a_max=1) * mask[..., None] * 255.

        cv2.imwrite(f'{args.orient_dir}/{basename}.png', orientation_map.astype('uint8'))
        np.save(f'{args.conf_dir}/{basename}.npy', confidence_map.astype('float16'))
        cv2.imwrite(f'{args.filtered_img_dir}/{basename}.png', filtered_img.astype('uint8'))
        cv2.imwrite(f'{args.vis_img_dir}/{basename}.png', vis_img.astype('uint8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--img_path', default='./implicit-hair-data/data/h3ds/00141/image/', type=str)
    parser.add_argument('--mask_path', default='./implicit-hair-data/data/h3ds/00141/image/', type=str)
    parser.add_argument('--orient_dir', default='./implicit-hair-data/data/h3ds/00141/orientation_maps/', type=str)
    parser.add_argument('--conf_dir', default='./implicit-hair-data/data/h3ds/00141/confidence_maps/', type=str)
    parser.add_argument('--filtered_img_dir', default='./implicit-hair-data/data/h3ds/00141/filtered_imgs/', type=str)
    parser.add_argument('--vis_img_dir', default='./implicit-hair-data/data/h3ds/00141/vis_imgs/', type=str)
    parser.add_argument('--dog_low', default=0.4, type=float)
    parser.add_argument('--dog_high', default=10, type=float)
    parser.add_argument('--num_frequencies', default=1, type=int)
    parser.add_argument('--num_filters', default=180, type=int)
    parser.add_argument('--num_sigmas_x', default=1, type=int)
    parser.add_argument('--num_sigmas_y', default=1, type=int)
    parser.add_argument('--num_offsets', default=1, type=int)
    parser.add_argument('--crop_size', default=-1, type=int)
    parser.add_argument('--patch_size', default=64, type=int)

    args, _ = parser.parse_known_args()

    main(args)