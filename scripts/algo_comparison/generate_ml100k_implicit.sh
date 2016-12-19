#!/usr/bin/env bash
# generate the implicit holdout split
python ../split_dataset.py \
../../data/ml100k/ratings.csv \
../../data/ml100k/implicit_holdout/train.csv \
../../data/ml100k/implicit_holdout/test.csv \
--make_implicit --implicit_th 4 --holdout_perc 0.8 --header 0 --rnd_seed 1