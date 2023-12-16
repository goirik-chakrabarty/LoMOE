#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python invert/src/inversion.py  \
        --input_image "sample/add/init_image.jpg" \
        --results_folder "output/add"

CUDA_VISIBLE_DEVICES=2 python src/inversion.py  \
        --input_image "/home/prathosh/aditya/DirectInversion/sample/style/init_image.jpg" \
        --results_folder "output/style"

CUDA_VISIBLE_DEVICES=2 python src/inversion.py  \
        --input_image "/home/prathosh/aditya/DirectInversion/sample/add/init_image.jpg" \
        --results_folder "output/add"


CUDA_VISIBLE_DEVICES=0 python invert/src/inversion.py  \
        --input_image "sample/style/init_image.jpg" \
        --results_folder "output/style"