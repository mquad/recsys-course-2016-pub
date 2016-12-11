#!/bin/bash

python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --recommender FunkSVD \
--params num_factors=20,lrate=0.01,reg=0.015