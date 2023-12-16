import os
from tqdm import tqdm
import json

with open("/home/prathosh/aditya/DirectInversion/dataDI_ALL.json", 'r') as fp:
    our_json = json.load(fp)

for seed in [0, 66, 512, 1234, 10]:
    for image in tqdm(our_json): 
        if not os.path.exists('diffedit_out/{}_{}.png'.format(image, str(seed))):
            os.system("python diffedit.py --out_path 'diffedit_out/{}_{}.png' --prompt '{}' --prompt_edit '{}' --origin_image '{}' --mask_path '{}' --seed {}".format(image, str(seed), our_json[image]['mask_prompt'], our_json[image]['fg_prompt'], our_json[image]['image_path'], our_json[image]['mask_path'], seed))
        break
    break