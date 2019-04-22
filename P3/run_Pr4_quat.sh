#!/usr/bin/env bash

# generate samples  ---  VAE

python main.py --mode test --model VAE --batch_size 8 --load_model

python main.py --mode gen --model VAE --load_model




# generate samples  ---  GAN

python main.py --mode test --model GAN --batch_size 8 --load_model

python main.py --mode gen --model GAN --load_model



# compute fid score

python -i score_fid.py ./sample/VAE/

python -i score_fid.py ./sample/GAN/

