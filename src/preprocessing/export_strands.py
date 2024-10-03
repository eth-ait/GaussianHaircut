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
import trimesh
from pysdf import SDF
from plyfile import PlyData, PlyElement



L = 100

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    return l


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--flame_mesh_dir', default='', type=str) 
    parser.add_argument('--hair_conf_path', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--iter', default='30000', type=str)
    args, _ = parser.parse_known_args()

    # Configuration of hair strands
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        strands_config = yaml.load(replaced_conf, Loader=yaml.Loader)

    strands_generator = OptimizableTexturedStrands(
        **strands_config['textured_strands'], 
        diffusion_cfg=strands_config['diffusion_prior'],
        data_dir=args.data_dir,
        flame_mesh_dir=args.flame_mesh_dir,
        num_guiding_strands=strands_config['extra_args']['num_guiding_strands']
    ).cuda()

    weights = torch.load(f'{args.data_dir}/strands_reconstruction/{args.model_name}/checkpoints/{args.iter}.pth')
    strands_generator.load_state_dict(weights[0][2])

    with torch.no_grad():
        p = strands_generator.forward_inference(30_000)[0]

    os.makedirs(f'{args.data_dir}/strands_reconstruction/{args.model_name}/strands', exist_ok=True)
    p_npy = p.cpu().numpy()

    mesh = trimesh.load(f'{args.flame_mesh_dir}/stage_3/mesh_final.obj')
    sdf_handle = SDF(mesh.vertices, mesh.faces)

    sdf = sdf_handle(p_npy.reshape(-1, 3))
    mask = (sdf.reshape(-1, L) < 0).mean(axis=1) >= 0.5
    print(f'Pruning {sum(~mask)} strands that intersect the head mesh')

    p_npy = p_npy[mask]
    with open(f'{args.data_dir}/strands_reconstruction/{args.model_name}/strands/{args.iter}_strands.pkl', 'wb') as f:
        pkl.dump(p_npy, f)

    xyz = p_npy.reshape(-1, 3)
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals), axis=1)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(f'{args.data_dir}/strands_reconstruction/{args.model_name}/strands/{args.iter}_strands.ply')