#!/bin/bash

python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --recommender item_knn --params similarity=pearson,k=50,shrinkage=100