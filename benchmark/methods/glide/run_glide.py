import os
from tqdm import tqdm
import json

with open("/home/prathosh/aditya/DirectInversion/dataDI_ALL.json", 'r') as fp:
    our_json = json.load(fp)

for seed in [0, 10, 66, 512, 1234]:
    for image in tqdm(our_json): 
        if not os.path.exists('glide_outputs_ALL/{}_{}.png'.format(image, str(seed))):
            # print("python glide.py --out_path 'glide_outputs_ALL/{}_{}.png' --prompt '{}' --image_path '/home/prathosh/goirik/PIE-Bench_v1/annotation_images/all_images/{}.jpg' --mask_path '{}' --seed {}".format(image, str(seed), our_json[image]["fg_prompt"], image, our_json[image]["mask_path"], int(seed)))
            os.system("python glide.py --out_path 'glide_out/{}_{}.png' --prompt '{}' --image_path '/home/prathosh/goirik/PIE-Bench_v1/annotation_images/all_images/{}.jpg' --mask_path '{}' --seed {}".format(image, str(seed), our_json[image]["fg_prompt"], image, our_json[image]["mask_path"], int(seed)))
        break
    break