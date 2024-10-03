from glob import glob
import os
from argparse import ArgumentParser
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import pickle as pkl
import math
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


def pil_loader(img_np):
    img = Image.fromarray(img_np)
    return img.convert('RGB')


def main(args):
    data_path = args.data_path
    os.makedirs(f'{data_path}/input', exist_ok=True)
    print(data_path)

    if os.path.exists(f'{data_path}/raw'):
        iqa_scores = {}
        for filename in tqdm(glob(f'{data_path}/raw/*')):
            basename = os.path.basename(filename).split('.')[0]
            img = Image.fromarray(cv2.imread(filename))
            img = np.asarray(torchvision.transforms.Resize(2160)(img))

            pred_scores = []

            for _ in range(10):
                img_tr = transforms(Image.fromarray(img))
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

            iqa_scores[f'{basename}.png'] = np.mean(pred_scores)

            cv2.imwrite(f'{data_path}/input/{basename}.png', img)
        pkl.dump(iqa_scores, open(f'{data_path}/iqa_scores.pkl', 'wb'))

    elif os.path.exists(f'{data_path}/raw.mp4'):
        vid = cv2.VideoCapture(f'{data_path}/raw.mp4')
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = int(vid.get(cv2.CAP_PROP_FPS))
        if args.target_fps:
            target_fps = args.target_fps
        else:
            target_fps = int(math.ceil(256 / length * source_fps))
        print(f'Extracting video with FPS {source_fps} into frames with FPS {target_fps}')

        iqa_scores = {}
        scores = []
        frames_rgb = []
        frames_idx = []

        for i in tqdm(range(1, length+1)):
            _, frame = vid.read()
            # if i % args.stride:
            #     continue
            frames_rgb.append(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = pil_loader(frame_rgb)

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

            scores.append(np.mean(pred_scores))
            frames_idx.append(i)

            if not (i % (source_fps // target_fps)):
                idx = np.argmax(np.asarray(scores))
                j = frames_idx[idx]
                cv2.imwrite(f'{data_path}/input/{j:06d}.png', frames_rgb[idx].astype('uint8'))
                iqa_scores[f'{j:06d}.png'] = scores[idx]
                scores = []
                frames_idx = []
                frames_rgb = []

        pkl.dump(iqa_scores, open(f'{data_path}/iqa_scores.pkl', 'wb'))

    else:
        raise


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--target_fps', default=0, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)