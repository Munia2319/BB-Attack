#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=2

# Run STBA attack with full parameter list
python main.py \
  --config /home/mmunia/stba-black-box/configs/default.yaml\
  --model_name Zhang2019Theoretically \
  --model_type robust \
  --max_images 10 \
  --lr 0.01 \
  --lambda_ 5 \
  --sigma 0.05 \
  --qmax 10000 \
  --nsample 20 \
  --adjustnum 3
