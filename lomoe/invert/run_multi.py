import os
from tqdm import tqdm
import json

all_files_dict = {}

with open('/home/prathosh/goirik/LOMOE_v2/text_prompts.txt') as file:
    lines = [line.rstrip() for line in file]

for folder in tqdm(os.listdir('/home/prathosh/goirik/LOMOE_v2/multi-obj')):
    all_files_dict[folder.zfill(2)] = {}

    image_path = os.path.join('/home/prathosh/goirik/LOMOE_v2/multi-obj', folder, 'init_image.png')
    all_files_dict[folder.zfill(2)]["image_path"] = image_path

    # if not folder.zfill(2) in os.listdir('our_outputs/MULTI'):
    #     os.system("python src/inversion.py --input_image {} --results_folder our_outputs/MULTI/{}".format(image_path, folder.zfill(2)))

    all_files_dict[folder.zfill(2)]["latent_path"] = "our_outputs/MULTI/{}/inversion/init_image.pt".format(folder.zfill(2))
    all_files_dict[folder.zfill(2)]["latent_list_path"] = "our_outputs/MULTI/{}/latentlist/init_image.pt".format(folder.zfill(2))
    all_files_dict[folder.zfill(2)]["bg_path"] = "our_outputs/MULTI/{}/prompt/init_image.txt".format(folder.zfill(2))
    all_files_dict[folder.zfill(2)]["bg_neg_path"] = "our_outputs/MULTI/{}/prompt/init_image.txt".format(folder.zfill(2))

    all_files_dict[folder.zfill(2)]["fg_prompt"] = lines[int(folder)]
    temp = ['"artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"'] * len(lines[int(folder)].split('" "'))
    all_files_dict[folder.zfill(2)]["fg_neg"] = ' '.join(temp)
    all_files_dict[folder.zfill(2)]["num_fgmasks"] = len(lines[int(folder)].split('" "'))

    masks = []
    for f in sorted(os.listdir(os.path.join('/home/prathosh/goirik/LOMOE_v2/multi-obj', folder))):
        if 'mask' in f:
            masks.append(os.path.join('/home/prathosh/goirik/LOMOE_v2/multi-obj/', folder, f))

    all_files_dict[folder.zfill(2)]["mask_path"] = '"' + '" "'.join(masks) + '"'

with open('dataDI_LOMOE_v2.json', 'w') as fp:
    json.dump(all_files_dict, fp)