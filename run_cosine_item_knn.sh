#!/bin/bash

python main.py data/ml100k/ratings.csv --header 0 --recommender item_knn --params similarity=cosine,k=50,shrinkage=100