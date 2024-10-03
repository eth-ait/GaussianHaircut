from glob import glob
import os
from argparse import ArgumentParser
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import Resize
import torchvision
import pickle as pkl
import sys
sys.path.append('../../ext/hyperIQA')
import models



transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 288)),
    torchvision.transforms.RandomCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])

model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# load our pre-trained model on the koniq-10k dataset
model_hyper.load_state_dict((torch.load('../../ext/hyperIQA/pretrained/koniq_pretrained.pkl')))


def main(args):
    data_path = args.data_path
    iqa_scores = {}
    
    for filename in tqdm(glob(f'{data_path}/images/*')):
        basename = os.path.basename(filename).split('.')[0]

        img = np.asarray(Image.open(f'{data_path}/images/{basename}.png'))
        mask_hair = np.asarray(Image.open(f'{data_path}/masks/hair/{basename}.png'))
        mask_face = np.asarray(Image.open(f'{data_path}/masks/face/{basename}.png'))
        mask_body = np.asarray(Image.open(f'{data_path}/masks/body/{basename}.png'))

        # Check intersection between face and hair
        if ((mask_hair > 127) * (mask_face > 127)).sum() > (mask_body > 127).sum() * 0.1:
            print(f'Skipping frame {basename}')
            continue

        # Crop
        h, w = img.shape[:2]
        i, j = np.nonzero(mask_hair > 0.0)
        l, r = j.min(), j.max()
        u, d = i.min(), i.max()
        sx = r - l
        sy = d - u
        px = int(sx * 0.05)
        py = int(sy * 0.05)
        l = max(l - px, 0)
        r = min(r + px, w)
        u = max(u - py, 0)
        d = min(d + py, h)
        
        img = img[u:d, l:r] * (mask_hair[u:d, l:r, None] / 255.)
        img = Image.fromarray(img.astype('uint8'))

        pred_scores = []

        for _ in range(10):
            img_tr = transforms(img)
            img_tr = torch.tensor(img_tr.cuda()).unsqueeze(0)
            with torch.no_grad():
                params = model_hyper(img_tr)

            # Building target network
            model_target = models.TargetNet(params).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(params['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            pred_scores.append(float(pred.item()))

        iqa_score_basename = np.mean(pred_scores)
        if iqa_score_basename > args.iqa_threshold:
            iqa_scores[basename] = iqa_score_basename

    pkl.dump(iqa_scores, open(f'{data_path}/iqa_scores_hair.pkl', 'wb'))
    # iqa_scores = pkl.load(open(f'{data_path}/iqa_scores_hair.pkl', 'rb'))
    
    # Split IQA scores into bins according to the frame index
    img_names = sorted(iqa_scores.keys())
    frame_idx = np.asarray([int(k) for k in img_names])
    print(frame_idx)
    num_bins = args.max_imgs
    while True:
        print(f'Trying to split into {num_bins}')
        hist, bins = np.histogram(frame_idx, bins=num_bins)
        print(hist)
        if sum(hist != 0) >= args.max_imgs:
            break
        else:
            num_bins += 1
    print(f'Splitting frames in {num_bins} bins')
    img_names_split = []
    for i in range(num_bins):
        if hist[i]:
            frame_idx_bin = frame_idx[np.logical_and(frame_idx >= bins[i], frame_idx < bins[i + 1])]
            img_names_chunk = []
            for j in frame_idx_bin:
                img_names_chunk.append('%06d.png' % j)
            img_names_split.append(img_names_chunk)
    assert len(img_names_split) >= args.max_imgs
    print(img_names_split)

    img_names_filtered = []
    for img_names_chunk in img_names_split:
        iqa_scores_chunk = []
        for img_name in img_names_chunk:
            iqa_scores_chunk.append(iqa_scores[img_name.replace('.png', '')])
        img_names_filtered.append(img_names_chunk[np.argmax(np.asarray(iqa_scores_chunk))])
    
    pkl.dump(img_names_filtered, open(f'{data_path}/iqa_filtered_names.pkl', 'wb'))

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--max_imgs', default=128, type=int)
    parser.add_argument('--iqa_threshold', default=50, type=float)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)