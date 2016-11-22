#!/bin/bash

python cv_eval.py ../data/ml100k/ratings.csv --header 0 --make_implicit --implicit_th 4 \
--recommender SLIM_mt --params l2_penalty=0.1,l1_penalty=0.001

python cv_eval.py ../data/ml100k/ratings.csv --header 0 --make_implicit --implicit_th 4 \
--recommender item_knn --params similarity=cosine,k=50,shrinkage=25,sparse_weights=True