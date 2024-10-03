from glob import glob
from tqdm import tqdm
import os
from argparse import ArgumentParser
from PIL import Image
import torch
import numpy as np
import cv2
import pickle as pkl
import math
# from torchvision.transforms import Resize, InterpolationMode



def main(args):
    data_path = args.data_path
    if os.path.exists(f'{data_path}/iqa_filtered_names.pkl'):
        img_names = pkl.load(open(f'{data_path}/iqa_filtered_names.pkl', 'rb'))
    else:
        img_names = os.listdir(f'{data_path}/images')

    os.makedirs(f'{data_path}/images_2', exist_ok=True)
    os.makedirs(f'{data_path}/images_4', exist_ok=True)
    os.makedirs(f'{data_path}/masks_2/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks_2/body', exist_ok=True)
    os.makedirs(f'{data_path}/masks_4/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks_4/body', exist_ok=True)

    # crop_params = {}

    for img_name in tqdm(img_names):
        basename = img_name.split('.')[0]
        img = np.asarray(Image.open(f'{data_path}/images/{img_name}'))
        h_old, w_old = img.shape[:2]
        mask_hair = np.asarray(Image.open(f'{data_path}/masks/hair/{basename}.png'))
        mask_body = np.asarray(Image.open(f'{data_path}/masks/body/{basename}.png'))
        if os.path.exists(f'{data_path}/masks/face/{basename}.png'):
            mask_face = np.asarray(Image.open(f'{data_path}/masks/face/{basename}.png'))
            # Check intersection between face and hair
            if ((mask_hair > 127) * (mask_face > 127)).sum() > (mask_body > 127).sum() * 0.1:
                print(f'Skipping frame {img_name}')
                continue

        # mask_binary = np.logical_or(mask_hair > 0, mask_face > 0) * 255
        # ks = int(max(h_old, w_old) * 0.03)
        # mask_binary = cv2.morphologyEx(mask_binary.astype('uint8'), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3 * ks, 3 * ks)))
        # mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3 * ks, 3 * ks)))

        # i, j = np.nonzero(mask_binary)

        # # Remove empty frames
        # if len(i) < int(h_old * w_old * 0.1):
        #     print(f'Skipping frame {img_name}')
        #     continue

        # Trim some of the body mask to focus the reconstruction process on the hair
        # mask_body = (mask_body * (mask_binary == 255)).astype('uint8')

        # i_min = i.min()
        # i_max = i.max()
        # j_min = j.min()
        # j_max = j.max()

        # l = j_min
        # r = j_max
        # u = i_min
        # d = i_max
        
        # cx = w_old / 2
        # cy = h_old / 2
        
        # print(cx - l, r - cx)
        # print(cy - u, d - cy)
        # print(l, u, r, d)
        
        # l = l if cx - l >= r - cx else w_old - r
        # r = r if r - cx >= cx - l else w_old - l
        # u = u if cy - u >= d - cy else h_old - d
        # d = d if d - cy >= cy - u else h_old - u
        
        # print(l, u, r, d)

        # h_crop = d - u
        # w_crop = r - l

        # img_crop = Image.fromarray(img).crop((l, u, r, d))
        # mask_hair_crop = Image.fromarray(mask_hair).crop((l, u, r, d))
        # mask_body_crop = Image.fromarray(mask_body).crop((l, u, r, d))

        # if h_crop == w_crop:
        #     img_crop.resize((args.img_size, args.img_size), Image.BICUBIC)
        #     mask_hair_crop.resize((args.img_size, args.img_size), Image.BICUBIC)
        #     mask_body_crop.resize((args.img_size, args.img_size), Image.BICUBIC)
        # else:
        #     img_crop = Resize(args.img_size - 1, InterpolationMode.BICUBIC, args.img_size)(img_crop)
        #     mask_hair_crop = Resize(args.img_size - 1, InterpolationMode.BICUBIC, args.img_size)(mask_hair_crop)
        #     mask_body_crop = Resize(args.img_size - 1, InterpolationMode.BICUBIC, args.img_size)(mask_body_crop)
        
        # w_new, h_new = img_crop.size

        Image.fromarray(img).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/images_2/{img_name}')
        Image.fromarray(img).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/images_4/{img_name}')
        Image.fromarray(mask_hair).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/masks_2/hair/{img_name}')
        Image.fromarray(mask_body).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/masks_2/body/{img_name}')
        Image.fromarray(mask_hair).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/masks_4/hair/{img_name}')
        Image.fromarray(mask_body).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/masks_4/body/{img_name}')

        # img_crop.save(f'{data_path}/images_crop/{img_name}')
        # mask_hair_crop.save(f'{data_path}/masks_crop/hair/{img_name}')
        # mask_body_crop.save(f'{data_path}/masks_crop/body/{img_name}')

        # crop_params[basename] = (cx, cy, w_crop, h_crop, w_new, h_new)

    # pkl.dump(crop_params, open(f'{data_path}/crop_params.pkl', 'wb'))

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    # parser.add_argument('--img_size', default=1024, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)