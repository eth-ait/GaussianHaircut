from glob import glob
import os
from argparse import ArgumentParser
from PIL import Image
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm



def main(args):
    param_paths = sorted(glob(f'{args.data_path}/pixie/*/*_param.pkl'))
    with open(f'{args.data_path}/initialization_pixie', 'wb') as f:
        for param_path in tqdm(param_paths):
            pkl.dump(pkl.load(open(param_path, 'rb')), f)

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)