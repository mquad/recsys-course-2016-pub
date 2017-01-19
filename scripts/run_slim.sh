#!/bin/bash

#python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
#--recommender SLIM_mt --params l2_penalty=0.1,l1_penalty=0.001

#python cv_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
#--recommender SLIM_mt --params l2_penalty=0.1,l1_penalty=0.001

python new_user_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
--n_observed 3 --recommender SLIM_mt --params l2_penalty=0.1,l1_penalty=0.001