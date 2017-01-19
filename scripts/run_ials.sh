#!/bin/bash

python main.py \
../data/ml100k/binary_holdout/train.csv \
../data/ml100k/binary_holdout/test.csv \
--header 0 --recommender IALS_np --is_binary \
--params num_factors=20,reg=10,iters=10,alpha=20,init_mean=0.0,init_std=0.1,rnd_seed=42