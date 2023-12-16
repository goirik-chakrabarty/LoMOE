import os
from tqdm import tqdm
import json

with open("/home/prathosh/aditya/DirectInversion/dataDI_ALL.json", 'r') as fp:
# with open("/home/prathosh/aditya/DirectInversion/dataDI_LOMOE_v2.json", 'r') as fp:
    our_json = json.load(fp)
    
OUT_DIR = 'bld_out_multi'

os.makedirs(OUT_DIR, exist_ok=True)

for seed in [0, 10, 66, 1234, 512]:
    for image in tqdm(our_json): 
        if not os.path.exists('{}/{}_{}.png'.format(OUT_DIR, image, str(seed))):
            mask_paths = our_json[image]["mask_path"][1:-1].split("\" \"")
            fg_prompts = our_json[image]["fg_prompt"][1:-1].split("\" \"")
            os.makedirs(f'{OUT_DIR}/{image}', exist_ok=True)
            for i, (mp, fp) in enumerate(zip(mask_paths, fg_prompts)):
                out_path = '{}/{}_{}.png'.format(OUT_DIR, image, str(seed))
                if i == 0:
                    os.system("python scripts/text_editing_stable_diffusion.py --output_path '{}/{}/{}_{}_{}.png' --prompt '{}' --init_image '{}' --mask '{}' --seed {}".format(OUT_DIR, image, image, str(seed), str(i), fp, our_json[image]["image_path"], mp, int(seed)))
                else:
                    os.system("python scripts/text_editing_stable_diffusion.py --output_path '{}/{}/{}_{}_{}.png' --prompt '{}' --init_image '{}/{}/{}_{}_{}.png' --mask '{}' --seed {}".format(OUT_DIR, image, image, str(seed), i, fp, OUT_DIR, image, image, str(seed), str(i-1), mp, int(seed)))
        break
    break   