import torch
from PIL import Image
import numpy as np

from tqdm import tqdm
import pickle

import os

import warnings
warnings.filterwarnings("ignore")

import argparse

import hpsv2
import json

import ImageReward as RM
reward_model = RM.load("ImageReward-v1.0")

import torch
from model import preprocess, load_model
from transformers import CLIPModel, CLIPProcessor

MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(MODEL)
vision_model = model.vision_model
vision_model.to(DEVICE)
del model
clip_processor = CLIPProcessor.from_pretrained(MODEL)

rating_model = load_model("../utils/aesthetics_scorer_rating_openclip_vit_h_14.pth").to(DEVICE)

def predict(img):
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
    return rating.detach().cpu().item()

#--------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("--folder_name", required=True, type=str)
args = parser.parse_args()

print("Metrics for folder : ", args.folder_name)

all_hps = []
all_ir = []
all_aesthetic = []

with open('../../data/LoMOE-Bench/LoMOE.json') as f:
    editing_prompts = json.load(f)
    
def get_bg_mask(mask_path, device, size):
    masks = mask_path[1:-1].split("\" \"")
    final_mask = np.array(Image.open(masks[0]).resize(size))
    for m in masks[1:]:
        final_mask += np.array(Image.open(m).resize(size))
    return torch.tensor((255 - final_mask)/255.0, dtype=torch.float32).to(device)

with open("../../data/LoMOE-Bench/utils/target_prompts.txt", 'r') as f:
    img_tgt_prompts = f.read().splitlines()

for image_name in tqdm(os.listdir(args.folder_name), disable=False):
    if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".JPEG"):

        image_path = os.path.join(args.folder_name, image_name)
        tgt_prompt = img_tgt_prompts[int(image_name.split('_')[0].split('.')[0])]

        all_ir.append(reward_model.score(tgt_prompt, [image_path]))
        all_hps.append(hpsv2.score(image_path, tgt_prompt, hps_version="v2.1"))
        all_aesthetic.append(predict(Image.open(image_path)))

print("#---------------#")
print("Average HPS: {:.5f} +- {:.5f}".format(np.mean(all_hps), np.std(all_hps)))
print("Average IR: {:.5f} +- {:.5f}".format(np.mean(all_ir), np.std(all_ir)))
print("Average Aesthetic: {:.5f} +- {:.5f}".format(np.mean(all_aesthetic), np.std(all_aesthetic)))

with open('results/{}.pickle'.format(args.folder_name.split('/')[-1]), 'wb') as f:
    pickle.dump((all_ir, all_hps, all_aesthetic), f)
