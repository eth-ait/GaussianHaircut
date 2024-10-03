import sys
sys.path.append('../../ext/Matte-Anything')
from matte_anything import generate_trimap, generate_checkerboard_image, convert_pixels
from PIL import Image
import numpy as np
import torch
import glob
from matplotlib import pyplot as plt
import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision.ops import box_convert
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict, annotate as dino_annotate
import argparse
import pathlib
import pickle as pkl
from torchvision.transforms import Resize, InterpolationMode
import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 



parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--image_format', default='jpg', type=str)
parser.add_argument('--postfix', default='', type=str)
parser.add_argument('--img_size', default=-1, type=int)
parser.add_argument('--max_size', default=-1, type=int)
parser.add_argument('--kernel_size', default=10, type=int)

args, _ = parser.parse_known_args()
args = parser.parse_args()

img_size = args.img_size
max_size = args.max_size
data_dir = args.data_path
model_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), 'ext', 'Matte-Anything')

models = {
	'vit_h': f'{model_dir}/pretrained/sam_vit_h_4b8939.pth',
    'vit_b': f'{model_dir}/pretrained/sam_vit_b_01ec64.pth'
}

vitmatte_models = {
	'vit_b': f'{model_dir}/pretrained/ViTMatte_B_DIS.pth',
}

vitmatte_config = {
	'vit_b': f'{model_dir}/configs/matte_anything.py',
}

grounding_dino = {
    'config': f'{model_dir}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    'weight': f'{model_dir}/pretrained/groundingdino_swint_ogc.pth'
}

device = 'cuda'

def init_segment_anything(model_type):
    """
    Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
    """
    
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    predictor = SamPredictor(sam)

    return predictor

def init_vitmatte(model_type):
    """
    Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    """
    cfg = LazyConfig.load(vitmatte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

    return vitmatte

def run_inference(input_x, selected_points, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, fg_caption, 
                    tr_box_threshold, tr_text_threshold, tr_caption = "glass, lens, crystal, diamond, bubble, bulb, web, grid"):
    
    predictor.set_image(input_x)

    dino_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = dino_transform(Image.fromarray(input_x), None)
    
    if len(selected_points) != 0:
        points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
        labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
        transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
        print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
        point_coords=transformed_points.permute(1, 0, 2)
        point_labels=labels.permute(1, 0)
    else:
        transformed_points, labels = None, None
        point_coords, point_labels = None, None
    
    if fg_caption is not None and fg_caption != "": # This section has benefited from the contributions of neuromorph,thanks! 
        fg_boxes, logits, phrases = dino_predict(
            model=grounding_dino,
            image=image_transformed,
            caption=fg_caption,
            box_threshold=fg_box_threshold,
            text_threshold=fg_text_threshold,
            device=device)
        print(logits, phrases)
        if fg_boxes.shape[0] == 0:
            # no fg object detected
            transformed_boxes = None
        else:
            h, w, _ = input_x.shape
            fg_boxes = torch.Tensor(fg_boxes).to(device)
            fg_boxes = fg_boxes * torch.Tensor([w, h, w, h]).to(device)
            fg_boxes = box_convert(boxes=fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            transformed_boxes = predictor.transform.apply_boxes_torch(fg_boxes, input_x.shape[:2])
    else:
        transformed_boxes = None
                
    # predict segmentation according to the boxes
    masks, scores, logits = predictor.predict_torch(
        point_coords = point_coords,
        point_labels = point_labels,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    masks = masks.cpu().detach().numpy()
    mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
    for ann in masks:
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[ann[0] == True, i] = color_mask[i]
    img = input_x / 255 * 0.3 + mask_all * 0.7
    
    # generate alpha matte
    torch.cuda.empty_cache()
    mask = masks[0][0].astype(np.uint8)*255
    trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
    trimap[trimap==128] = 0.5
    trimap[trimap==255] = 1
    
    boxes, logits, phrases = dino_predict(
        model=grounding_dino,
        image=image_transformed,
        caption= tr_caption,
        box_threshold=tr_box_threshold,
        text_threshold=tr_text_threshold,
        device=device)
    annotated_frame = dino_annotate(image_source=input_x, boxes=boxes, logits=logits, phrases=phrases)
    
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    if boxes.shape[0] == 0:
        # no transparent object detected
        pass
    else:
        h, w, _ = input_x.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        trimap = convert_pixels(trimap, xyxy)

    input = {
        "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0)/255,
        "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
    }

    torch.cuda.empty_cache()
    alpha = vitmatte(input)['phas'].flatten(0,2)
    alpha = alpha.detach().cpu().numpy()
    
    # get a green background
    background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)

    # calculate foreground with alpha blending
    foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

    # calculate foreground with mask
    foreground_mask = input_x * np.expand_dims(mask/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(mask/255, axis=2).repeat(3,2))/255

    foreground_alpha[foreground_alpha>1] = 1
    foreground_mask[foreground_mask>1] = 1

    # return img, mask_all
    trimap[trimap==1] == 0.999

    return  mask, alpha, foreground_mask, foreground_alpha

os.makedirs(f'{data_dir}/masks{args.postfix}/hair', exist_ok=True)
os.makedirs(f'{data_dir}/masks{args.postfix}/body', exist_ok=True)
os.makedirs(f'{data_dir}/masks{args.postfix}/face', exist_ok=True)

sam_model = 'vit_h'
vitmatte_model = 'vit_b'

predictor = init_segment_anything(sam_model)
vitmatte = init_vitmatte(vitmatte_model)
grounding_dino = dino_load_model(grounding_dino['config'], grounding_dino['weight'])

img_names = os.listdir(f'{data_dir}/images{args.postfix}')
filenames = [f'{data_dir}/images{args.postfix}/{img_name}' for img_name in img_names]
# filenames = glob.glob(f'{data_dir}/images{args.postfix}/*')

for filename in tqdm.tqdm(sorted(filenames)):
    with torch.no_grad():
        img = Image.open(filename)
        orig_img_size = img.size
        if img_size != -1 or max_size != -1:
            img_size = max_size - 1 if img_size == -1 else img_size
            max_size = max_size if max_size != -1 else None
            img = Resize(img_size, InterpolationMode.BICUBIC, max_size)(img)

        _, mask_hair, _, _ = run_inference(
            np.asarray(img), [], 
            erode_kernel_size=args.kernel_size, 
            dilate_kernel_size=args.kernel_size, 
            fg_box_threshold=0.25, 
            fg_text_threshold=0.25, 
            fg_caption="hair", 
            tr_box_threshold=0.5, 
            tr_text_threshold=0.25,
            tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")

        _, mask_face, _, _ = run_inference(
            np.asarray(img), [], 
            erode_kernel_size=args.kernel_size, 
            dilate_kernel_size=args.kernel_size, 
            fg_box_threshold=0.5, # higher threshold to reduce false positive
            fg_text_threshold=0.25, 
            fg_caption="face", 
            tr_box_threshold=0.5, 
            tr_text_threshold=0.25,
            tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")

        _, mask_body, _, _ = run_inference(
            np.asarray(img), [], 
            erode_kernel_size=args.kernel_size, 
            dilate_kernel_size=args.kernel_size, 
            fg_box_threshold=0.25, 
            fg_text_threshold=0.25, 
            fg_caption="human", 
            tr_box_threshold=0.5, 
            tr_text_threshold=0.25,
            tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")
        
        mask_hair = Image.fromarray((mask_hair * 255).astype('uint8'))
        mask_face = Image.fromarray((mask_face * 255).astype('uint8'))
        mask_body = Image.fromarray((mask_body * 255).astype('uint8'))

        if img_size != -1:
            mask_hair = mask_hair.resize(orig_img_size, Image.BICUBIC)
            mask_face = mask_face.resize(orig_img_size, Image.BICUBIC)
            mask_body = mask_body.resize(orig_img_size, Image.BICUBIC)

        mask_hair.save(filename.replace(f'images{args.postfix}', f'masks{args.postfix}/hair').replace(args.image_format, 'png'))
        mask_face.save(filename.replace(f'images{args.postfix}', f'masks{args.postfix}/face').replace(args.image_format, 'png'))
        mask_body.save(filename.replace(f'images{args.postfix}', f'masks{args.postfix}/body').replace(args.image_format, 'png'))
