#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=2

# Run STBA attack with full parameter list
python main.py \
  --config /home/mmunia/BB-Attack/configs/default.yaml\
  --model_name Zhang2019Theoretically \
  --model_type robust \
  --max_images 10 \
  --lr 0.1 \
  --lambda_ 5 \
  --sigma 0.2 \
  --qmax 1000 \
  --nsample 10 \
  --adjustnum 20
