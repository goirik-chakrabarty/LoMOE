# Usage 

Put the image and the mask in `sample` with appropriate name then run the following:

## Inversion Step:

```
CUDA_VISIBLE_DEVICES=0 python invert/src/inversion.py  \
        --input_image "sample/add/init_image.jpg" \
        --results_folder "output/add"
```

## Edit Step:

```
CUDA_VISIBLE_DEVICES=2 python edit/main.py \
  --mask_paths "sample/add/mask_1.jpg" \
  --bg_prompt "output/add/prompt/init_image.txt" \
  --bg_negative "output/add/prompt/init_image.txt" \
  --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" \
  --H 512 \
  --W 512 \
  --bootstrapping 20 \
  --latent 'output/add/inversion/init_image.pt' \
  --latent_list 'output/add/latentlist/init_image.pt' \
  --rec_path 'out_rec.png' \
  --edit_path 'out.png' \
  --fg_prompts "a red dog collar" \
  --seed 1234 \
  --save_path 'output/add/merged.png'
```