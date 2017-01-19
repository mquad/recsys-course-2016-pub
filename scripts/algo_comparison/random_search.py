import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime as dt

from recpy.recommenders.item_knn import ItemKNNRecommender
from recpy.recommenders.slim import MultiThreadSLIM
from recpy.recommenders.mf import FunkSVD, IALS_numpy, BPRMF
from recpy.recommenders.non_personalized import TopPop, GlobalEffects

from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr
from recpy.utils.tuning import random_search_cv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def holdout_eval(recommender, train, test, at=10):
    # train the recommender
    logger.info('Recommender: {}'.format(recommender))
    tic = dt.now()
    logger.info('Training started')
    print(train.sum())
    recommender.fit(train)
    logger.info('Training completed in {}'.format(dt.now() - tic))
    # evaluate the ranking quality
    roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_eval = 0
    nusers = train.shape[0]
    for test_user in range(nusers):
        user_profile = train[test_user]
        relevant_items = test[test_user].indices
        if len(relevant_items) > 0:
            n_eval += 1
            # this will rank **all** items
            recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True)
            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_ += roc_auc(recommended_items, relevant_items)
            precision_ += precision(recommended_items, relevant_items, at=at)
            recall_ += recall(recommended_items, relevant_items, at=at)
            map_ += map(recommended_items, relevant_items, at=at)
            mrr_ += rr(recommended_items, relevant_items, at=at)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
    roc_auc_ /= n_eval
    precision_ /= n_eval
    recall_ /= n_eval
    map_ /= n_eval
    mrr_ /= n_eval
    ndcg_ /= n_eval
    return roc_auc_, precision_, recall_, map_, mrr_, ndcg_


metric = roc_auc
cv_folds = 5
at = 10
is_binary = True

train_df = read_dataset('../../data/ml100k/binary_holdout/train.csv', sep=',', header=0)
test_df = read_dataset('../../data/ml100k/binary_holdout/test.csv', sep=',', header=0)
nusers, nitems = train_df.user_idx.max() + 1, train_df.item_idx.max() + 1
train = df_to_csr(train_df, is_binary=is_binary, nrows=nusers, ncols=nitems)
test = df_to_csr(test_df, is_binary=is_binary, nrows=nusers, ncols=nitems)

#
# TopPop
#
# RecommenderClass = TopPop
# param_space = {}
# # Evaluate all the metrics over the hold out split
# recommender = RecommenderClass()
# metrics = holdout_eval(recommender, train, test, at=at)
# logger.info('Metrics: {}'.format(metrics))
#
#
# GlobalEffects
#
RecommenderClass = GlobalEffects
param_space = {
    'lambda_user': np.arange(10, 100, 20),
    'lambda_item': np.arange(10, 100, 20),
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))

#
# ItemKNNRecommender
#
RecommenderClass = ItemKNNRecommender
param_space = {
    'shrinkage': np.arange(0, 100, 20),
    'k': np.arange(50, 200, 50),
    'similarity': ['cosine']
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))

#
# SLIM
#
RecommenderClass = MultiThreadSLIM
param_space = {
    'l1_penalty': np.logspace(-4, 2, 5),
    'l2_penalty': np.logspace(-4, 2, 5),
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))

#
# FunkSVD
#
RecommenderClass = FunkSVD
param_space = {
    'num_factors': [20],
    'iters': [10],
    'lrate': np.logspace(-4, -1, 5),
    'reg': np.logspace(-3, 1, 5),
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))
#
#
# IALS_numpy
#
RecommenderClass = IALS_numpy
param_space = {
    'num_factors': [20],
    'iters': [10],
    'alpha': np.arange(20, 100, 20),
    'reg': np.logspace(-3, 1, 5),
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
best_config = {
    'num_factors': 20,
    'iters': 10,
    'alpha': 20,
    'reg': 10,
}
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))

#
# BPRMF
#
RecommenderClass = BPRMF
param_space = {
    'num_factors': [20],
    'iters': [10],
    'sample_with_replacement': [True],
    'lrate': np.logspace(-3, -1, 4),
    'user_reg': np.logspace(-3, 0, 4),
    'pos_reg': np.logspace(-3, 0, 4),
    # 'neg_reg': np.logspace(-4, 0, 5),
}
# Tune the hyper-parameters with RandomSearchCV
logger.info('Tuning {} with RandomSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = random_search_cv(RecommenderClass,
                                       train_df,
                                       param_space,
                                       metric=roc_auc,
                                       cv_folds=cv_folds,
                                       is_binary=is_binary,
                                       user_key='user_idx',
                                       item_key='item_idx')
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)
metrics = holdout_eval(recommender, train, test, at=at)
logger.info('Metrics: {}'.format(metrics))
