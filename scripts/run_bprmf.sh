#!/bin/bash

python main.py \
../data/ml100k/implicit_holdout/train.csv \
../data/ml100k/implicit_holdout/test.csv \
--header 0 --recommender BPRMF --is_implicit \
--params num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42