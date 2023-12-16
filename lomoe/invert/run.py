import os
from tqdm import tqdm
import json

all_files_dict = {}

with open("/home/prathosh/goirik/PIE-Bench_v1/mapping_file_our.json", 'r') as fp:
    our_json = json.load(fp)

for folder in os.listdir('/home/prathosh/goirik/PIE-Bench_v1/ours'):
    for img_name in tqdm(os.listdir(os.path.join('/home/prathosh/goirik/PIE-Bench_v1/ours', folder, 'images'))):
        all_files_dict[img_name.split('.')[0]] = {}
    
        image_path = os.path.join('/home/prathosh/goirik/PIE-Bench_v1/ours', folder, 'images', img_name)
        all_files_dict[img_name.split('.')[0]]["image_path"] = image_path
        mask_path = os.path.join('/home/prathosh/goirik/PIE-Bench_v1/ours', folder, 'masks', img_name)
        all_files_dict[img_name.split('.')[0]]["mask_path"] = mask_path

        # os.system("python src/inversion.py --input_image {} --results_folder our_outputs/{}/{}".format(image_path, folder, img_name.split('.')[0]))

        all_files_dict[img_name.split('.')[0]]["fg_prompt"] = our_json[img_name.split('.')[0]]["fg_prompt"]
        all_files_dict[img_name.split('.')[0]]["latent_path"] = "our_outputs/{}/{}/inversion/{}.pt".format(folder, img_name.split('.')[0], img_name.split('.')[0])
        all_files_dict[img_name.split('.')[0]]["latent_list_path"] = "our_outputs/{}/{}/latentlist/{}.pt".format(folder, img_name.split('.')[0], img_name.split('.')[0])
        all_files_dict[img_name.split('.')[0]]["bg_path"] = "our_outputs/{}/{}/prompt/{}.txt".format(folder, img_name.split('.')[0], img_name.split('.')[0])
        all_files_dict[img_name.split('.')[0]]["bg_neg_path"] = "our_outputs/{}/{}/prompt/{}.txt".format(folder, img_name.split('.')[0], img_name.split('.')[0])
        all_files_dict[img_name.split('.')[0]]["fg_neg"] = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"

with open('dataDI_ALL.json', 'w') as fp:
    json.dump(all_files_dict, fp)