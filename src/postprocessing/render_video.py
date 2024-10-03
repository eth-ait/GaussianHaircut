import os
import glob
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import pickle as pkl
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation, RotationSpline
import numpy
from PIL import Image, ImageOps
from pytorch3d.io import load_obj, save_ply
import cv2



# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations: 
#   - Oxford's visual geometry group matlab toolbox 
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)
    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K @ sg
    R = sg @ R
    # det(R) negative, just invert; the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
        R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:,-1])[0]
    T = -R @ C
    return K, R, T

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    
    return r, q

def get_camera_params(camera):
    K, R_world2cv, T_world2cv = KRT_from_P(camera)
    R_world2cv_quat = Rotation.from_matrix(R_world2cv).as_quat()
    
    return K, R_world2cv_quat, T_world2cv

def get_concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def main(blender_path, input_path, exp_name_1, exp_name_3, strand_length, speed_up, max_frames):
    os.makedirs(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/results', exist_ok=True)

    frames = [
        int(fname.split('.')[0])
        for fname in 
        sorted(os.listdir(f'{input_path}/images_2'))
    ]
    cameras = pkl.load(open(f'{input_path}/3d_gaussian_splatting/{exp_name_1}/cameras/30000_matrices.pkl', 'rb'))

    # Unpack cameras
    R = []
    K = []
    T = []
    for frame in frames:
        scale_x, scale_y = 1080, 1920
        intrinsics, pose = load_K_Rt_from_P(None, cameras['%06d' % frame].transpose(0, 1)[:3, :4].numpy())
        pose_all_inv = np.linalg.inv(pose)
        intrinsics_modified = intrinsics.copy()

        intrinsics_modified[0, 0] /= 2  # Halve fx
        intrinsics_modified[1, 1] /= 2  # Halve fy
        intrinsics_modified[0, 2] /= 2  # Halve cx
        intrinsics_modified[1, 2] /= 2  # Halve cy

        intrinsics_modified[0, 2] += 0.5  # Adjust cx
        intrinsics_modified[1, 2] += 0.5  # Adjust cy

        # Create a scaling matrix
        scaling_matrix = np.array([[scale_x, 0, 0, 0],
                                    [0, scale_y, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        projection_matrix = scaling_matrix @ intrinsics_modified @ pose_all_inv
                
        K_world2cv, R_world2cv, T_world2cv = KRT_from_P(projection_matrix[:3])
                
        R.append(R_world2cv)
        K.append(K_world2cv)
        T.append(T_world2cv)

    rotations = Rotation.from_matrix(np.stack(R))
    spline = RotationSpline(frames, rotations)
    R_interp = spline(list(range(frames[-1]))).as_matrix()

    cameras_interp = []
    prev_j = -1
    next_j = 0

    for i in range(frames[-1]):
        if i in frames:
            prev_j += 1
            next_j += 1

        prev_K = K[prev_j]
        prev_T = T[prev_j]

        next_K = K[next_j]
        next_T = T[next_j]

        alpha = 1 - (i - frames[prev_j]) / (frames[next_j] - frames[prev_j])

        K_cur = prev_K * alpha + next_K * (1 - alpha)
        T_cur = prev_T * alpha + next_T * (1 - alpha)

        cameras_interp.append(K_cur @ np.concatenate([R_interp[i], T_cur[:, None]], axis=1))
        
    cameras_interp = np.stack(cameras_interp)[frames[0]:frames[-1]:speed_up][:max_frames]
    np.save(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/cameras.npy', cameras_interp)

    verts, faces, _ = load_obj(f'{input_path}/flame_fitting/{exp_name_1}/stage_3/mesh_final.obj')
    save_ply(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/raw_head.ply', verts=verts, faces=faces.verts_idx)

    # Unpack head model (which also contains hair in case of neus and unisurf)
    head_ply = PlyData.read(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/raw_head.ply')

    # Apply blender transform to vertices of the head model
    head_vertex = (
        np.stack([
            head_ply.elements[0].data['x'], 
            head_ply.elements[0].data['y'], 
            head_ply.elements[0].data['z']], axis=1).reshape(-1, 3, 1)
    )[..., 0]
    head_vertex = [tuple(vertex) for vertex in head_vertex.tolist()]
    head_vertex = np.array(head_vertex, dtype=np.dtype('float, float, float'))
    head_vertex.dtype.names = ['x', 'y', 'z']

    head_new_ply = PlyData([PlyElement.describe(head_vertex, 'vertex'), head_ply.elements[1]])
    head_new_ply.write(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/head.ply')

    # Unpack strands (no transform is needed)
    strands_ply = PlyData.read(f'{input_path}/curves_reconstruction/{exp_name_3}/strands/10000_strands.ply').elements[0].data
    strands_npy = (
        np.stack([
            strands_ply['x'], 
            -strands_ply['z'], 
            strands_ply['y']], axis=1).reshape(-1, strand_length, 3, 1)
    )[..., 0]
    np.save(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/hair.npy', strands_npy)

    os.system(
        f'{blender_path} -b main.blend -P render_color.py -- --args ' \
        f'{input_path}/curves_reconstruction/{exp_name_3}/blender/cameras.npy ' \
        f'{input_path}/curves_reconstruction/{exp_name_3}/blender/head.ply ' \
        f'{input_path}/curves_reconstruction/{exp_name_3}/blender/hair.npy ' \
        f'{input_path}/curves_reconstruction/{exp_name_3}/blender/results ' \
        f'128 {frames[0]} {speed_up}'
    )


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--blender_path', default='/home/ezakharov/Libraries/blender-3.6.11-linux-x64/blender', type=str)
    parser.add_argument('--input_path', default='/home/ezakharov/Datasets/hair_reconstruction/NeuralHaircut/jenya', type=str)
    parser.add_argument('--exp_name_1', default='stage1_lor=0.1', type=str)
    parser.add_argument('--exp_name_3', default='stage3_lor=0.1', type=str)
    parser.add_argument('--strand_length', default=100, type=int)
    parser.add_argument('--speed_up', default=4, type=int)
    parser.add_argument('--max_frames', default=200, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args.blender_path, args.input_path, args.exp_name_1, args.exp_name_3, args.strand_length, args.speed_up, args.max_frames)