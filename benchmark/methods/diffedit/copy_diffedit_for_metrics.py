import os
import glob
from PIL import Image

from tqdm import tqdm

files = sorted(glob.glob('/home/prathosh/goirik/DiffEdit-stable-diffusion/diffedit_out/*.png'))

ct = 0

if not os.path.exists('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_0'):
    os.mkdir('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_0')
    os.mkdir('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_10')
    os.mkdir('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_66')
    os.mkdir('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_512')
    os.mkdir('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_1234')

for idx in tqdm(range(0, len(files), 5)):
    image_names = [x.split('/')[-1].split('_')[0] for x in [files[idx], files[idx + 1], files[idx + 2], files[idx + 3], files[idx + 4]]]
    images = [Image.open(x) for x in [files[idx], files[idx + 1], files[idx + 2], files[idx + 3], files[idx + 4]]]

    if len(set(image_names)) == 1:
        if files[idx].split('/')[-1].split('_')[0] in [y[:-4] for y in os.listdir('/home/prathosh/aditya/benchmark/testbench/images')]:
            images[0].save('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_0/{}.png'.format(files[idx].split('/')[-1].split('_')[0]))
            images[1].save('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_10/{}.png'.format(files[idx].split('/')[-1].split('_')[0]))
            images[4].save('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_66/{}.png'.format(files[idx].split('/')[-1].split('_')[0]))
            images[3].save('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_512/{}.png'.format(files[idx].split('/')[-1].split('_')[0]))
            images[2].save('/home/prathosh/aditya/benchmark/testbench/2_diffedit/diffedit_1234/{}.png'.format(files[idx].split('/')[-1].split('_')[0]))