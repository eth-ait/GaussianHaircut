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
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
from tqdm import tqdm
import shutil



def main(input_path, exp_name_3):
    os.makedirs(f'{input_path}/curves_reconstruction/{exp_name_3}/raw_frames', exist_ok=True)
    os.system(f'ffmpeg -i {input_path}/raw.mp4 -qscale:v 2 {input_path}/curves_reconstruction/{exp_name_3}/raw_frames/%06d.jpg')
    os.makedirs(f'{input_path}/curves_reconstruction/{exp_name_3}/frames', exist_ok=True)
    img_names = sorted(os.listdir(f'{input_path}/curves_reconstruction/{exp_name_3}/train/ours_30000/renders'))
    print(len(img_names))
    for i, img_name in tqdm(enumerate(img_names)):
        img_basename = img_name.split('.')[0]
        render_3dgs = Image.open(f'{input_path}/curves_reconstruction/{exp_name_3}/train/ours_30000/renders/{img_basename}.png').convert('RGB')
        render_blender = Image.open(f'{input_path}/curves_reconstruction/{exp_name_3}/blender/results/{img_basename}.png')
        render_blender_new = Image.new("RGBA", render_blender.size, "WHITE")
        render_blender_new.paste(render_blender, (0, 0), render_blender)
        render_blender = render_blender_new.convert('RGB')
        gt = Image.open(f'{input_path}/curves_reconstruction/{exp_name_3}/raw_frames/%06d.jpg' % (int(img_basename) - 1)).convert('RGB')
        w, h = render_3dgs.size
        render_blender_resized = Resize(h, interpolation=InterpolationMode.BICUBIC)(render_blender)
        render_blender_cropped_resized = CenterCrop((h, w))(render_blender_resized)
        gt_resized = Resize(w, interpolation=InterpolationMode.BICUBIC)(gt)
        frame = Image.fromarray(np.concatenate([np.asarray(gt_resized), np.asarray(render_blender_cropped_resized), np.asarray(render_3dgs)], axis=1))
        frame_resized = Resize(720, interpolation=InterpolationMode.BICUBIC)(frame)
        frame_resized.save(f'{input_path}/curves_reconstruction/{exp_name_3}/frames/%06d.png' % i)
    os.system(f'ffmpeg -r 30 -i "{input_path}/curves_reconstruction/{exp_name_3}/frames/%06d.png" -c:v libx264 -vb 20M {input_path}/curves_reconstruction/{exp_name_3}/vis.mp4')
    
    # Cleanup
    shutil.rmtree(f'{input_path}/curves_reconstruction/{exp_name_3}/frames')
    shutil.rmtree(f'{input_path}/curves_reconstruction/{exp_name_3}/raw_frames')

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--input_path', default='/home/ezakharov/Datasets/hair_reconstruction/NeuralHaircut/jenya', type=str)
    parser.add_argument('--exp_name_3', default='stage3_lor=0.1', type=str)

    args, _ = parser.parse_known_args()

    main(args.input_path, args.exp_name_3)