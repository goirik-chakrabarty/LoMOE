import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import lpips
import numpy as np

from tqdm import tqdm
import pickle

import os

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import warnings
warnings.filterwarnings("ignore")

import argparse

model_name = "openai/clip-vit-base-patch16"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(model_name, device=device)
model = CLIPModel.from_pretrained(model_name).to(device)

#--------------------------------------------------------------#

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

#--------------------------------------------------------------#

from clip_transforms import Global_crops, dino_structure_transforms, dino_texture_transforms
from torchvision import transforms
from clip_extractor import VitExtractor
import torch.nn.functional as F
from torchvision.transforms import Resize

structure_transforms = dino_structure_transforms
texture_transforms = dino_texture_transforms
base_transform = transforms.Compose([
    transforms.ToTensor(),
])

global_A_patches = transforms.Compose(
    [
        structure_transforms,
        Global_crops(n_crops=1, min_cover=0.95, last_transform=base_transform)
    ]
)

global_B_patches = transforms.Compose(
    [
        texture_transforms,
        Global_crops(n_crops=1, min_cover=0.95, last_transform=base_transform)
    ]
)

def calculate_global_ssim_loss(outputs, inputs):
    loss = 0.0

    extractor = VitExtractor(model_name='dino_vitb8', device=device)

    imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    global_resize_transform = Resize(224, max_size=480)

    global_transform = transforms.Compose([global_resize_transform,
                                                imagenet_norm
                                                ])

    for a, b in zip(inputs, outputs):  # avoid memory limitations
        a = global_transform(a)
        b = global_transform(b)
        with torch.no_grad():
            target_keys_self_sim = extractor.get_keys_self_sim_from_input(a.unsqueeze(0).to(device), layer_num=11)
        keys_ssim = extractor.get_keys_self_sim_from_input(b.unsqueeze(0).to(device), layer_num=11)
        loss += F.mse_loss(keys_ssim, target_keys_self_sim)
    return loss

#--------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("--folder_name", required=True, type=str)
args = parser.parse_args()

print("Metrics for folder : ", args.folder_name)

all_clip = []
all_tgtclip = []
all_bglpips = []
all_bgmse = []
all_bgpsnr = []
all_bgssim = []

all_struct = []

import json

with open('../../data/LoSOE-Bench/LoSOE.json') as f:
    editing_prompts = json.load(f)

for image_name in tqdm(os.listdir(args.folder_name), disable=True):
    if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".JPEG"):
        if len(image_name.split('.')[0]) == 12: 
            IMAGE_NAME = image_name.split('.')[0]
        else:
            IMAGE_NAME = image_name.split('_')[0]
            
        original = Image.open(os.path.join('images', IMAGE_NAME + '.jpg'))
            
        mask = torch.tensor((255 - np.array(Image.open(os.path.join('masks', IMAGE_NAME + '.jpg')).resize(original.size)))/255.0, dtype=torch.float32).to(device)

        image_path = os.path.join(args.folder_name, image_name)
        image = Image.open(image_path)
        inputs = processor(text=None, images=image, return_tensors="pt").to(device)

        caption = open(os.path.join('prompts', IMAGE_NAME + '.txt')).read().strip()
        text_inputs = processor(caption, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

        similarity_score = (image_features @ text_features.T).mean()
        all_clip.append(similarity_score.item())

        tgt_prompt = editing_prompts[IMAGE_NAME]["editing_prompt"].replace("[", "").replace("]", "")
        text_inputs = processor(tgt_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

        similarity_score = (image_features @ text_features.T).mean()
        all_tgtclip.append(similarity_score.item())

        #--------------------------------------------------------------#

        img0 = torch.tensor(2 * (np.array(original)/255.0) - 1).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float32).to(device)
        img1 = torch.tensor(2 * (np.array(image)/255.0) - 1).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float32).to(device)

        d = loss_fn_alex(torch.mul(mask, img0), torch.mul(mask, img1))
        all_bglpips.append(d.item())

        #--------------------------------------------------------------#

        A = original.convert('RGB')
        B = image.convert('RGB')

        sample = {}

        sample['A_global'] = global_A_patches(A)
        sample['B_global'] = global_B_patches(B)

        all_struct.append(calculate_global_ssim_loss(sample['A_global'], sample['B_global']).detach().cpu())

        #---------------------------------------------------------------#

        mask_arr = (255 - np.array(Image.open(os.path.join('masks', IMAGE_NAME + '.jpg')).resize(original.size)))/255.0
        mask_arr = np.dstack((mask_arr, mask_arr, mask_arr))
        original_arr = np.uint8(np.multiply(np.array(original), mask_arr))
        image_arr = np.uint8(np.multiply(np.array(image), mask_arr))

        all_bgmse.append(((original_arr - image_arr)**2).mean())
        all_bgpsnr.append(psnr(original_arr, image_arr))
        all_bgssim.append(ssim(original_arr, image_arr, channel_axis = 2))


print("#---------------#")
print("Average SRC CLIP Score : {:.5f} +- {:.5f}".format(np.mean(all_clip), np.std(all_clip)))
print("Average TGT CLIP Score : {:.5f} +- {:.5f}".format(np.mean(all_tgtclip), np.std(all_tgtclip)))
print("Average BG LPIPS : {:.5f} +- {:.5f}".format(np.mean(all_bglpips), np.std(all_bglpips)))
print("Average BG PSNR : {:.5f} +- {:.5f}".format(np.mean(all_bgpsnr), np.std(all_bgpsnr)))
print("Average BG MSE : {:.5f} +- {:.5f}".format(np.mean(all_bgmse), np.std(all_bgmse)))
print("Average BG SSIM : {:.5f} +- {:.5f}".format(np.mean(all_bgssim), np.std(all_bgssim)))
print("Average Structure Distance : {:.5f} +- {:.5f}".format(np.mean(all_struct), np.std(all_struct)))

with open('{}_metrics.pickle'.format(args.folder_name), 'wb') as f:
    pickle.dump((all_clip, all_tgtclip, all_bglpips, all_bgpsnr, all_bgmse, all_bgssim, all_struct), f)
