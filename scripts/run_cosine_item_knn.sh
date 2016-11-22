#!/bin/bash

python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --make_implicit --implicit_th 4 \
 --recommender item_knn --params similarity=cosine,k=50,shrinkage=25