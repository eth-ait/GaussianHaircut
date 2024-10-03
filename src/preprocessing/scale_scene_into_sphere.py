import numpy as np
import pickle
import numpy as np
import torch
import argparse
import os
import sys
from copy import deepcopy
from argparse import ArgumentParser
sys.path.append('../')
sys.path.append('../../ext/NeuralHaircut')
sys.path.append('../../ext/NeuralHaircut/k-diffusion')
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel



def prune_points(gaussians, mask):
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._label = gaussians._label[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._orient_conf = gaussians._orient_conf[mask]
    return gaussians


def main(args, dataset):
    path = os.path.join(args.model_path, 'point_cloud', f'iteration_{args.iter}', 'raw_point_cloud.ply')
    os.makedirs(os.path.join(args.model_path, 'point_cloud_cropped', f'iteration_{args.iter}'), exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)

    gaussians.load_ply(path)

    labels = gaussians.get_label
    opacities = gaussians.get_opacity

    mask_hair = torch.logical_and((labels >= 0.5)[:, 0], (opacities >= 0.5)[:, 0])

    gaussians_hair = deepcopy(gaussians)
    gaussians_hair = prune_points(gaussians_hair, mask_hair)

    xyz_hair = gaussians_hair.get_xyz
    tr = torch.zeros(3).cuda()

    for _ in range(5):
        norm_hair = torch.linalg.norm(xyz_hair - tr, dim=-1)
        treshold = torch.median(norm_hair, dim=0).values * 5
        print(treshold)

        xyz_hair = xyz_hair[norm_hair < treshold]

        tr = xyz_hair.mean(axis=0)
        s = norm_hair[norm_hair < treshold].max()
        print(tr, s)

    xyz = gaussians.get_xyz
    norm = torch.linalg.norm(xyz - tr, dim=-1)
    mask = norm < s
    gaussians = prune_points(gaussians, mask)
    gaussians.save_ply(f"{args.model_path}/point_cloud_cropped/iteration_{args.iter}/point_cloud.ply")

    d = {'scale': s.item(),
        'translation': [el.item() for el in list(tr)]}

    with open(os.path.join(args.path_to_data, 'scale.pickle'), 'wb') as f:
        pickle.dump(d, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--path_to_data', default='', type=str) 
    parser.add_argument('--iter', default=30_000, type=int) 
    args = get_combined_args(parser)

    dataset = model.extract(args)

    main(args, dataset)