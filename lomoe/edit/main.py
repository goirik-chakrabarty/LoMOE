from utils import prep_unet
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

logging.set_verbosity_error()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float16)

def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        # elif self.sd_version=='ip':
        #     model_key = "stabilityai/stable-diffusion-2-inpainting"
        # elif self.sd_version=='ip_1.5':
        #     model_key = "runwayml/stable-diffusion-inpainting"
        else:
            model_key = self.sd_version

        self.vae = AutoencoderKL.from_pretrained(
            model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet").to(self.device)

        self.unet = prep_unet(self.unet)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler")

        self.d_ref_t2attn = {}  
        self.image_latent_ref = {} 

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_random_background(self, n_samples):
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def reconstruct(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, latent_path=None, latent_list_path=None, num_fgmasks=2):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)

        latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        noise_loss_list=[]
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            count.zero_()
            value.zero_()

            for h_start, h_end, w_start, w_end in views:
                latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(len(prompts), 1, 1, 1)

                latent_model_input = torch.cat([latent_view] * 2).type(torch.cuda.HalfTensor)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                self.d_ref_t2attn[t.item()] = {}
                self.image_latent_ref[t.item()] = {}
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "CrossAttention" and 'attn2' in name:
                        attn_mask = module.attn_probs
                        attn_mask = torch.cat(tuple([attn_mask] * num_fgmasks), dim=0)
                        self.d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                if latent_list is not None:
                    noise_loss_list.append(latent_list[-2-i] - latents_view_denoised)
                    latents_view_denoised = latents_view_denoised + noise_loss_list[-1]

            latent = latents_view_denoised
            self.image_latent_ref[t.item()] = latent.detach().cpu()

        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor))
        img = T.ToPILImage()(imgs[0].cpu())
        return img, noise_loss_list

    def generate(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, ca_coef=1, seg_coef=0.25, noise_loss_list=None, latent_path=None, latent_list_path=None):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)

        latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        mask_tensor = masks[0]
        mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = torch.Tensor(np.array([np.array(mask_tensor.cpu())] * 4)).to(device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            count.zero_()
            value.zero_()

            masks_view = masks
            latent_view = latent.repeat(len(prompts), 1, 1, 1)
            if i < bootstrapping:
                bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                bg = self.scheduler.add_noise(bg, noise, t)
                latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (1 - masks_view[1:])

            latent_model_input = torch.cat([latent_view] * 2).type(torch.cuda.HalfTensor)

            x_in = latent_model_input.detach().clone()
            x_in.requires_grad = True

            opt = torch.optim.SGD([x_in], lr=0.1)

            noise_pred = self.unet(x_in, t, encoder_hidden_states=text_embeds.detach())['sample']

            loss = 0.0
            loss_ca = 0.0
            loss_seg = 0.0
            for name, module in self.unet.named_modules():
                module_name = type(module).__name__
                if module_name == "CrossAttention" and 'attn2' in name:
                    curr = module.attn_probs
                    ref = self.d_ref_t2attn[t.item()][name].detach().to(device)
                    loss_ca += ((curr-ref)**2).sum((1, 2)).mean(0)

            latents = x_in.chunk(2)[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_view_denoised = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            if noise_loss_list is not None:
                latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True) # 00:57
            latent = torch.where(count > 0, latent / count, latent) # 00:57

            latent_cur = latent.squeeze(0)
            latent_ref = self.image_latent_ref[t.item()].detach().to(device).squeeze(0)

            loss_seg += (torch.multiply(mask_tensor, latent_cur - latent_ref)**2).sum((1, 2)).mean(0)

            torch.cuda.empty_cache()
            loss = ca_coef * loss_ca + seg_coef * loss_seg
            loss.backward(retain_graph=False)
            opt.step()

            with torch.no_grad():
                noise_pred = self.unet(x_in.detach(), t, encoder_hidden_states=text_embeds)['sample']

            latents = x_in.detach().chunk(2)[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_view_denoised = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            if noise_loss_list is not None:
                latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True)
            latent = torch.where(count > 0, latent / count, latent)

        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor)) 
        img = T.ToPILImage()(imgs[0].cpu())
        return img


def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float16) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_paths', nargs='+')
    parser.add_argument('--rec_path', type=str)
    parser.add_argument('--edit_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--bg_prompt', type=str)
    parser.add_argument('--bg_negative', type=str)
    parser.add_argument('--fg_prompts', nargs='+')
    parser.add_argument('--fg_negative', nargs='+')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.4', '1.5', '2.0', 'ip'], help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bootstrapping', type=int, default=20)
    parser.add_argument('--num_fgmasks', type=int, default=1)
    parser.add_argument('--latent', type=str)
    parser.add_argument('--latent_list', type=str)
    parser.add_argument('--ca_coef', type=float, default=1.0)
    parser.add_argument('--seg_coef', type=float, default=1.75)

    opt = parser.parse_args()

    device = torch.device('cuda')

    ca_coef = opt.ca_coef #1.0
    seg_coef = opt.seg_coef #1.75

    print(ca_coef, seg_coef)

    seed = opt.seed

    seed_everything(seed)

    sd = MultiDiffusion(device, opt.sd_version)

    seed_everything(seed)

    fg_masks = torch.zeros((1, opt.H//8, opt.W//8))
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    masks = torch.cat([bg_mask, fg_masks])

    prompts = [open(opt.bg_prompt).read().strip()]
    neg_prompts = [open(opt.bg_negative).read().strip()]

    rec_img, noise_loss_list = sd.reconstruct(masks, prompts, neg_prompts, opt.H, opt.W, opt.steps, bootstrapping=opt.bootstrapping, latent_path=opt.latent, latent_list_path=opt.latent_list, num_fgmasks=opt.num_fgmasks+1)
    rec_img.save(opt.rec_path)

    fg_masks = torch.cat([preprocess_mask(mask_path, opt.H // 8, opt.W // 8, device) for mask_path in opt.mask_paths])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    prompts = [open(opt.bg_prompt).read().strip()] + opt.fg_prompts
    neg_prompts = [open(opt.bg_negative).read().strip()] + opt.fg_negative

    img = sd.generate(masks, prompts, neg_prompts, opt.H, opt.W, opt.steps, bootstrapping=opt.bootstrapping, ca_coef=ca_coef, seg_coef=seg_coef, noise_loss_list=noise_loss_list, latent_path=opt.latent, latent_list_path=opt.latent_list)

    img.save(opt.edit_path)

    images = [rec_img, img]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    if opt.save_path:
        new_im.save(opt.save_path)    
        
# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=silver_o=llama_s=real/mask_1.png" --bg_prompt "a statue of a llama on display in a museum" --bg_negative "a statue of a llama on display in a museum" --fg_prompts "a silver giraffe" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/9/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/9/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --seed 66
# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/10/mask_1.jpg" --bg_prompt "a cartoon fox driving a toy car" --bg_negative "a cartoon fox driving a toy car" --fg_prompts "small eyes" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/10/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/10/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --seed 66

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/add/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/add/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/add/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/add/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/add/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "a red dog collar" --seed 66 --save_path '1_add.png'

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/background/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/background/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/background/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 45 --latent '/home/prathosh/aditya/DirectInversion/output/background/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/background/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "a starry night background" --seed 66 --save_path '2_background.png'

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/change/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/change/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/change/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/change/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/change/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "a big pink yarn ball" --seed 66 --save_path '3_change.png'

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/color/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/color/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/color/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 25 --latent '/home/prathosh/aditya/DirectInversion/output/color/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/color/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "a blue lamb" --seed 66 --save_path '4_color.png'

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/material/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/material/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/material/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/material/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/material/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "bronze horse" --seed 66 --save_path '5_material.png'

# python region_based_wml_efficient_direct.py --mask_paths "/home/prathosh/aditya/DirectInversion/assets/dataset/style/mask_1.jpg" --bg_prompt "/home/prathosh/aditya/DirectInversion/output/style/prompt/init_image.txt" --bg_negative "/home/prathosh/aditya/DirectInversion/output/style/prompt/init_image.txt" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20 --latent '/home/prathosh/aditya/DirectInversion/output/style/inversion/init_image.pt' --latent_list '/home/prathosh/aditya/DirectInversion/output/style/latentlist/init_image.pt' --rec_path 'out_rec.png' --edit_path 'out.png' --fg_prompts "a sketch of a cat" --seed 66 --save_path '6_style.png'

####################################### Choosen ###################################################
# Pig
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=black_o=pig_s=outdoor/mask_1.png" --bg_prompt "a small black pig standing on top of a red surface" --bg_negative "a small black pig standing on top of a red surface" --fg_prompts "a black and brown wild boar" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 30

# Cat figurine
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=square_o=cat_s=real/mask_1.png" --bg_prompt "a cat figurine sitting on top of a plate" --bg_negative "a cat figurine sitting on top of a plate" --fg_prompts "a dog figurine" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 10
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=square_o=cat_s=real/mask_1.png" --bg_prompt "a cat figurine sitting on top of a plate" --bg_negative "a cat figurine sitting on top of a plate" --fg_prompts "a dog figurine" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 10

# Group of horse - a=group_o=horse_s=indoor [not reconstructing]
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=group_o=horse_s=indoor/mask_2.png" --bg_prompt "a group of horses standing in a gym" --bg_negative "a group of horses standing in a gym" --fg_prompts "a zebra" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# Octopus - a=gray_o=octopus_s=outdoor
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=gray_o=octopus_s=outdoor/mask_1.png" --bg_prompt "an octopus figurine sitting on a wooden deck" --bg_negative "an octopus figurine sitting on a wooden deck" --fg_prompts "a colorful octopus with red dots" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# Metal Cat - a=metal_o=cat_s=outdoor
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=metal_o=cat_s=outdoor/mask_1.png" --bg_prompt "a silver cat statue sitting in the middle of a field" --bg_negative "a silver cat statue sitting in the middle of a field" --fg_prompts "a metal dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# Group of cats - a=orange_o=cat_s=real
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=orange_o=cat_s=real/mask_1.png" --bg_prompt "a group of cats standing around a tray of cupcakes" --bg_negative "a group of cats standing around a tray of cupcakes" --fg_prompts "an orange dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=orange_o=cat_s=real/mask_2.png" --bg_prompt "a group of cats standing around a tray of cupcakes" --bg_negative "a group of cats standing around a tray of cupcakes" --fg_prompts "an orange dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=orange_o=cat_s=real/mask_3.png" --bg_prompt "a group of cats standing around a tray of cupcakes" --bg_negative "a group of cats standing around a tray of cupcakes" --fg_prompts "an orange dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=orange_o=cat_s=real/mask_3.png" --bg_prompt "a group of cats standing around a tray of cupcakes" --bg_negative "a group of cats standing around a tray of cupcakes" --fg_prompts "a real orangish dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20


# Metal llama - a=silver_o=llama_s=real
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/a=silver_o=llama_s=real/mask_1.png" --bg_prompt "a statue of a llama on display in a museum" --bg_negative "a statue of a llama on display in a museum" --fg_prompts "a silver giraffe" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# Bowl of fruit - ILSVRC2012_val_00000023
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/ILSVRC2012_val_00000023/mask_1.png" --bg_prompt "a group of green apples with stickers on them" --bg_negative "a group of green apples with stickers on them" --fg_prompts "an orange" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/ILSVRC2012_val_00000023/mask_1.png, /home/prathosh/aditya/benchmark/dino/masks/ILSVRC2012_val_00000023/mask_2.png" --bg_prompt "a group of green apples with stickers on them" --bg_negative "a group of green apples with stickers on them" --fg_prompts "an orange colored orange, a red pomegranate" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/ILSVRC2012_val_00000023/mask_1.png" --bg_prompt "a group of green apples with stickers on them" --bg_negative "a group of green apples with stickers on them" --fg_prompts "a red pomegranate" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/ILSVRC2012_val_00000023/mask_1.png" --bg_prompt "a group of green apples with stickers on them" --bg_negative "a group of green apples with stickers on them" --fg_prompts "a red apple" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# flickr_dog_000045
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_dog_000045/mask_1.png" --bg_prompt "a close up of a dog wearing a pink sweater" --bg_negative "a close up of a dog wearing a pink sweater" --fg_prompts "a smiling dog" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_dog_000045/mask_1.png" --bg_prompt "a close up of a dog wearing a pink sweater" --bg_negative "a close up of a dog wearing a pink sweater" --fg_prompts "a close up of a sad dog wearing a pink sweater" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_dog_000045/mask_1.png" --bg_prompt "a close up of a dog wearing a pink sweater" --bg_negative "a close up of a dog wearing a pink sweater" --fg_prompts "a cat with sunglasses" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_dog_000045/mask_1.png" --bg_prompt "a close up of a dog wearing a pink sweater" --bg_negative "a close up of a dog wearing a pink sweater" --fg_prompts "a cat with sunglasses and a party hat" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_dog_000045/mask_1.png" --bg_prompt "a close up of a dog wearing a pink sweater" --bg_negative "a close up of a dog wearing a pink sweater" --fg_prompts "a cat with sunglasses and a party hat on the head" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20

# flickr_wild_000504
# python region_based_wml_efficient.py --mask_paths "/home/prathosh/aditya/benchmark/dino/masks/flickr_wild_000504/mask_1.png" --bg_prompt "a grey wolf staring at the camera in the snow" --bg_negative "a grey wolf staring at the camera in the snow" --fg_prompts "a leopard" --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" --H 512 --W 512 --bootstrapping 20