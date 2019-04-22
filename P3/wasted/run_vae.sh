#!/usr/bin/env bash

#python main.py --mode train --model VAE --num_epochs 30 --batch_size 128 --test_interval 10 --save_model

python main.py --mode test --model VAE --batch_size 8 --load_model

python main.py --mode gen --model VAE --load_model