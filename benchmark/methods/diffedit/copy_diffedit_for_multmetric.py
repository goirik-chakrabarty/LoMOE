import os
import glob
from PIL import Image

from tqdm import tqdm

IN_DIR = '/home/prathosh/goirik/DiffEdit-stable-diffusion/diffedit_out_multi'
N_SEEDS = 5

folders = [os.path.join(IN_DIR,x) for x in sorted(os.listdir(IN_DIR))]

if not os.path.exists('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_0'):
    os.mkdir('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_0')
    os.mkdir('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_10')
    os.mkdir('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_66')
    os.mkdir('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_512')
    os.mkdir('/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_1234')

for image_folder in tqdm(folders):
    n_mask = int(len(os.listdir(image_folder))/N_SEEDS - 1)
    image_names = []
    for path in sorted(os.listdir(image_folder)):
        if str(n_mask) in path.split('_')[-1]:
            image_names.append(f'{image_folder}/{path}')

    images = [Image.open(x) for x in image_names]

    bname = image_folder.split('/')[-1]

    if len(image_names) == 5:
        images[0].save(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_0/{bname}.png')
        images[1].save(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_10/{bname}.png')
        images[4].save(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_66/{bname}.png')
        images[3].save(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_512/{bname}.png')
        images[2].save(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_1234/{bname}.png')
    
        # print(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_0/{bname}.png')
        # print(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_10/{bname}.png')
        # print(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_66/{bname}.png')
        # print(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_512/{bname}.png')
        # print(f'/home/prathosh/aditya/benchmark/multi_testbench/2_diffedit/diffedit_1234/{bname}.png')
        # print()