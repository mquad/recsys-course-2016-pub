#!/bin/bash

python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --recommender AsySVD \
--params num_factors=20,lrate=0.01,reg=0.015,iters=10