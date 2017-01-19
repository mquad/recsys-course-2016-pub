#!/bin/bash

#python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
# --recommender item_knn --params similarity=cosine,k=50,shrinkage=25,normalize=False

python new_user_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
--n_observed 3 --recommender item_knn --params similarity=cosine,k=50,shrinkage=25,normalize=False
