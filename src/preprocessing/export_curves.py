from argparse import ArgumentParser
import yaml
import torch
import os
import pickle as pkl
import sys
sys.path.append('../')
sys.path.append('../../ext/NeuralHaircut')
sys.path.append('../../ext/NeuralHaircut/k-diffusion')
from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from scene import Scene
from scipy.spatial.transform import Rotation, RotationSpline
import numpy
import numpy as np
from scene import GaussianModelCurves
import trimesh
from pysdf import SDF
from plyfile import PlyData, PlyElement
import tqdm



L = 100

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    return l

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--flame_mesh_path', default='', type=str) 
    parser.add_argument('--scalp_mesh_path', default='', type=str) 
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--iter', default='30000', type=str)
    args, _ = parser.parse_known_args()

    mesh = trimesh.load(args.flame_mesh_path)
    # scalp = trimesh.load(args.scalp_mesh_path)
    sdf_handle = SDF(mesh.vertices, mesh.faces)

    pts = torch.load(f'{args.data_dir}/curves_reconstruction/{args.model_name}/checkpoints/{args.iter}.pth')[0][0]
    pts = pts.reshape(-1, L, 3)
    p_npy = pts.detach().cpu().numpy()
    print(f'Processing {p_npy.shape[0]} strands')

    # sdf = sdf_handle(p_npy.reshape(-1, 3)).reshape(-1, L)
    # mask_prune = (sdf < 0).mean(axis=1) < 0.5
    # print(f'Pruning {sum(mask_prune)} strands that intersect the head mesh')
    # sdf = sdf[~mask_prune]
    # p_npy = p_npy[~mask_prune]

    # mask_split = (sdf < 0).mean(axis=1) < 1.0
    # print(f'Splitting {sum(mask_split)} strands that partially intersect the head mesh')
    # p_npy_to_split = p_npy[mask_split]
    # sdf_to_split = sdf[mask_split]

    # p_npy_split = []
    # split_mask = sdf_to_split > 0
    # for i in tqdm.tqdm(range(p_npy_to_split.shape[0])):
    #     strand_i = p_npy_to_split[i]
    #     split_mask_i = split_mask[i].tolist()
    #     skip_next = split_mask_i[0]
    #     while len(split_mask_i) > 1:
    #         if True in split_mask_i: # next point inside FLAME mesh
    #             j = split_mask_i.index(True)
    #         else:
    #             j = len(split_mask_i) + 1
    #         if not skip_next and j > 1:
    #             strand_j = strand_i[:j]
    #             strand_j = np.concatenate([
    #                 np.stack([
    #                     np.linspace(strand_j[0, 0], strand_j[1, 0], 100 - strand_j.shape[0] + 1),
    #                     np.linspace(strand_j[0, 1], strand_j[1, 1], 100 - strand_j.shape[0] + 1),
    #                     np.linspace(strand_j[0, 2], strand_j[1, 2], 100 - strand_j.shape[0] + 1)], 
    #                     axis=-1
    #                 ),
    #                 strand_j[1:]
    #             ],
    #             axis=0)
    #             p_npy_split.append(strand_j)
    #         else:
    #             skip_next = False
    #         strand_i = strand_i[j:]
    #         split_mask_i = split_mask_i[j:]
    #         if False in split_mask_i: # next point outside FLAME mesh
    #             k = split_mask_i.index(False)
    #             strand_i = strand_i[k:]
    #             split_mask_i = split_mask_i[k:]
    #             # Find a closest point on the scalp surface and attach it to the strand
    #             origin, dist, _ = trimesh.proximity.closest_point(scalp, strand_i[:1])
    #             if dist[0] < 0.1:
    #                 strand_i = np.concatenate([origin, strand_i], axis=0)
    #             else:
    #                 # Skip to the next strand segment
    #                 skip_next = True
    #         else:
    #             break
    # p_npy_split = np.stack(p_npy_split, axis=0)
    # p_npy = np.concatenate([p_npy[~mask_split], p_npy_split], axis=0)

    print(f'Saving {p_npy.shape[0]} strands')

    os.makedirs(f'{args.data_dir}/curves_reconstruction/{args.model_name}/strands', exist_ok=True)
    with open(f'{args.data_dir}/curves_reconstruction/{args.model_name}/strands/{args.iter}_strands.pkl', 'wb') as f:
        pkl.dump(p_npy, f)

    xyz = p_npy.reshape(-1, 3)
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals), axis=1)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(f'{args.data_dir}/curves_reconstruction/{args.model_name}/strands/{args.iter}_strands.ply')