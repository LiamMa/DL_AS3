#!/usr/bin/env bash

# python main.py --mode train --model GAN --batch_size 256 --num_epochs 300 --test_interval 30 --G_update_interval 2 --save_model

python main.py --mode test --model GAN --batch_size 8 --load_model

python main.py --mode gen --model GAN --load_model