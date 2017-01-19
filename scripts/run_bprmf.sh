#!/bin/bash


# USER UNIFORM ITEM UNIFORM SAMPLING

python main.py \
../data/ml100k/binary_holdout/train.csv \
../data/ml100k/binary_holdout/test.csv \
--header 0 --recommender BPRMF --is_binary \
--params num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,\
sample_with_replacement=True,sampling_type=user_uniform_item_uniform,\
init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42


# USER UNIFORM ITEM POPULAR SAMPLING
# use factor sampling_pop_alpha to control the importance of popularity
# (0.0 -> uniform sampling - 1.0 -> fully pop sampling)
python main.py \
../data/ml100k/binary_holdout/train.csv \
../data/ml100k/binary_holdout/test.csv \
--header 0 --recommender BPRMF --is_binary \
--params num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,\
sample_with_replacement=True,sampling_type=user_uniform_item_pop,sampling_pop_alpha=0.75,\
init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42