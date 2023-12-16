import os
from tqdm import tqdm
import json

all_files_dict = {}

with open("/home/prathosh/goirik/PIE-Bench_v1/PIE-Bench_v1.json", 'r') as fp:
    our_json = json.load(fp)

with open("/home/prathosh/goirik/PIE-Bench_v1/mask_prompts.json", 'r') as fp:
    mask_prompts_json = json.load(fp)

for img_name in tqdm(os.listdir('/home/prathosh/goirik/PIE-Bench_v1/annotation_images/all_images')):
    all_files_dict[img_name.split('.')[0]] = {}

    image_path = os.path.join('/home/prathosh/goirik/PIE-Bench_v1/annotation_images/all_images', img_name)
    all_files_dict[img_name.split('.')[0]]["image_path"] = image_path
    mask_path = os.path.join('/home/prathosh/goirik/PIE-Bench_v1/ours/all_masks', img_name)
    all_files_dict[img_name.split('.')[0]]["mask_path"] = mask_path

    # os.system("python src/inversion.py --input_image {} --results_folder our_outputs/ALL/{}".format(image_path, img_name.split('.')[0]))

    all_files_dict[img_name.split('.')[0]]["fg_prompt"] = our_json[img_name.split('.')[0]]["fg_prompt"]
    all_files_dict[img_name.split('.')[0]]["mask_prompt"] = mask_prompts_json[img_name.split('.')[0]]

    all_files_dict[img_name.split('.')[0]]["editing_instruction"] = our_json[img_name.split('.')[0]]["editing_instruction"]

    all_files_dict[img_name.split('.')[0]]["latent_path"] = "our_outputs/ALL/{}/inversion/{}.pt".format(img_name.split('.')[0], img_name.split('.')[0])
    all_files_dict[img_name.split('.')[0]]["latent_list_path"] = "our_outputs/ALL/{}/latentlist/{}.pt".format(img_name.split('.')[0], img_name.split('.')[0])
    all_files_dict[img_name.split('.')[0]]["bg_path"] = "our_outputs/ALL/{}/prompt/{}.txt".format(img_name.split('.')[0], img_name.split('.')[0])
    all_files_dict[img_name.split('.')[0]]["bg_neg_path"] = "our_outputs/ALL/{}/prompt/{}.txt".format(img_name.split('.')[0], img_name.split('.')[0])
    all_files_dict[img_name.split('.')[0]]["fg_neg"] = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"

print("Total keys : ", len(all_files_dict))

with open('dataDI_ALL.json', 'w') as fp:
    json.dump(all_files_dict, fp)