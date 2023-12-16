import os
from tqdm import tqdm
import json

with open("/home/prathosh/aditya/DirectInversion/dataDI_LOMOE_v2.json", 'r') as fp:
    our_json = json.load(fp)

OUT_DIR = 'glide_out_multi'

os.makedirs(OUT_DIR, exist_ok=True)

with open('/home/prathosh/goirik/LOMOE_v2/mask_orig_prompts.txt') as file:
    lines = [line.rstrip() for line in file]

for seed in [0, 10, 66, 1234, 512]:
    for image in tqdm(our_json):
        if not os.path.exists('{}/{}_{}.png'.format(OUT_DIR, image, str(seed))):
            mask_paths = our_json[image]["mask_path"][1:-1].split("\" \"")
            fg_prompts = our_json[image]["fg_prompt"][1:-1].split("\" \"")
            mask_prompts = lines[int(image)][1:-1].split("\" \"")
            
            os.makedirs(f'{OUT_DIR}/{image}', exist_ok=True)
            for i, (mpath, fprompt, mprompt) in enumerate(zip(mask_paths, fg_prompts, mask_prompts)):
                out_path = '{}/{}_{}.png'.format(OUT_DIR, image, str(seed))
                if i == 0:
                    os.system("python glide.py --out_path '{}/{}/{}_{}_{}.png' --prompt '{}' --image_path '{}' --mask_path '{}' --seed {}".format(
                        OUT_DIR, image, image, str(seed), str(i), fprompt, our_json[image]['image_path'], mpath, int(seed)))
                else:
                    os.system("python glide.py --out_path '{}/{}/{}_{}_{}.png' --prompt '{}' --image_path '{}/{}/{}_{}_{}.png' --mask_path '{}' --seed {}".format(
                        OUT_DIR, image, image, str(seed), str(i), fprompt, OUT_DIR, image, image, str(seed), str(i-1), mpath, int(seed)))
        break
    break
