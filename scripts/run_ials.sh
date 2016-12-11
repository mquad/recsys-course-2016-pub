#!/bin/bash

python holdout_eval.py ../data/ml100k/ratings.csv --header 0 \
--recommender IALS_np --params num_factors=20,reg=0.015