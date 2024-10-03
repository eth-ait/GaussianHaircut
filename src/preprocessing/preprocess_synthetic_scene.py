import os
from argparse import ArgumentParser
import torch
from torchvision.transforms import ToPILImage
import pickle as pkl
import shutil



def main(args):
    data_path = args.data_path
    os.makedirs(f'{data_path}/images', exist_ok=True)
    os.makedirs(f'{data_path}/masks/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks/body', exist_ok=True)
    os.makedirs(f'{data_path}/orientations/angles', exist_ok=True)
    os.makedirs(f'{data_path}/orientations/vars', exist_ok=True)
    os.makedirs(f'{data_path}/flame_fitting/scalp_data', exist_ok=True)
    os.makedirs(f'{data_path}/flame_fitting/stage_3', exist_ok=True)
    filenames = os.listdir(f'{data_path}/image')
    basenames = [filename.split('.')[0].split('_')[1] for filename in filenames]
    for basename in basenames:
        shutil.move(f'{data_path}/image/img_{basename}.png', f'{data_path}/images/{basename}.png')
        shutil.move(f'{data_path}/mask/img_{basename}.png', f'{data_path}/masks/body/{basename}.png')
        shutil.move(f'{data_path}/hair_mask/img_{basename}.png', f'{data_path}/masks/hair/{basename}.png')
        shutil.move(f'{data_path}/orientation_maps/img_{basename}.png', f'{data_path}/orientations/angles/{basename}.png')
        shutil.move(f'{data_path}/confidence_maps/img_{basename}.npy', f'{data_path}/orientations/vars/{basename}.npy')
    os.rmdir(f'{data_path}/image')
    os.rmdir(f'{data_path}/mask')
    os.rmdir(f'{data_path}/hair_mask')
    os.rmdir(f'{data_path}/orientation_maps')
    os.rmdir(f'{data_path}/confidence_maps')
    shutil.move(f'{data_path}/cut_scalp_verts.pickle', f'{data_path}/flame_fitting/scalp_data/cut_scalp_verts.pickle')
    dif_mask = 1 - torch.load(f'{data_path}/dif_mask.pth')
    ToPILImage()(dif_mask).save(f'{data_path}/flame_fitting/scalp_data/dif_mask.png')
    os.remove(f'{data_path}/dif_mask.pth')
    shutil.move(f'{data_path}/scaled_head_prior.obj', f'{data_path}/flame_fitting/stage_3/mesh_final.obj')

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--data_path', default='', type=str)
    args = parser.parse_args()

    main(args)