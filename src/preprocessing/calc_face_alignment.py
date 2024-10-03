import argparse
import face_alignment
import pickle as pkl
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from glob import glob
from PIL import Image
import numpy as np



fa2d = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

def main(args):
    data_path = args.data_path
    image_paths = sorted(glob(f'{data_path}/{args.image_dir}/*'))

    os.makedirs(f'{data_path}/face_alignment/vis_2d', exist_ok=True)
    os.makedirs(f'{data_path}/face_alignment/vis_3d', exist_ok=True)

    lmks_2d = {}
    lmks_3d = {}

    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        image_scaled_np = np.asarray(image)
        lmk_2d = fa2d.get_landmarks_from_image(image_scaled_np)
        lmk_3d = fa3d.get_landmarks_from_image(image_scaled_np)
        basename = os.path.basename(image_path).split('.')[0]

        if lmk_2d is not None and len(lmk_2d):
            plt.figure(figsize=(9, 16))
            plt.imshow(image_scaled_np)
            plt.scatter(lmk_2d[0][..., 0], lmk_2d[0][..., 1])
            plt.savefig(f'{data_path}/face_alignment/vis_2d/{basename}.jpg')
            lmks_2d[basename] = lmk_2d[0]

        if lmk_3d is not None and len(lmk_3d):
            plt.figure(figsize=(9, 16))
            plt.imshow(image)
            plt.scatter(lmk_3d[0][..., 0], lmk_3d[0][..., 1])
            plt.savefig(f'{data_path}/face_alignment/vis_3d/{basename}.jpg')
            lmks_3d[basename] = lmk_3d[0]

    pkl.dump(lmks_2d, open(f'{data_path}/face_alignment/lmks_2d.pkl', 'wb'))
    pkl.dump(lmks_3d, open(f'{data_path}/face_alignment/lmks_3d.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--image_dir', default='', type=str)
    args = parser.parse_args()
    
    main(args)