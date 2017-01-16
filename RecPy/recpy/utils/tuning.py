import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr
from recpy.utils.data_utils import df_to_csr
from recpy.utils.split import k_fold_cv

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def grid_search_cv(RecommenderClass, dataset, param_space, metric=roc_auc, at=None,
                   cv_folds=5, is_implicit=True, user_key='user_id', item_key='item_id', rating_key='rating',
                   rnd_seed=1234):
    """
    Finds the best hyper-parameters of a recommender algorithm with Grid Search

    :param RecommenderClass: Class of the recommender to tune (must be subclass of Recommender)
    :param dataset: data to use for tuning
    :param param_space: space of the parameters to explore
    :param metric: metric to maximize
    :param at: optional length of the recommendation list used in recommendaiton
    :param cv_folds: number of cross-validation iters
    :param is_implicit: True to discard ratings, False otherwise
    :param user_key: name of the column with user ids in dataset
    :param item_key: name of the column with item ids in dataset
    :param rating_key: name of the column with ratings in dataset
    :param rnd_seed: random seed used for cross-validation
    :return: a tuple with (best configuration, best metric value)
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} cv trials)'.format(space_size, space_size * cv_folds))
    param_grid = ParameterGrid(param_space)
    # compute the cv splits
    nusers, nitems = dataset[user_key].max() + 1, dataset[item_key].max() + 1
    cv_split = []
    for train_df, test_df in k_fold_cv(dataset,
                                       user_key=user_key,
                                       item_key=item_key,
                                       k=cv_folds,
                                       clean_test=True,
                                       seed=rnd_seed):
        train = df_to_csr(train_df, is_implicit=is_implicit, nrows=nusers, ncols=nitems,
                          user_key=user_key, item_key=item_key, rating_key=rating_key)
        test = df_to_csr(test_df, is_implicit=is_implicit, nrows=nusers, ncols=nitems,
                         user_key=user_key, item_key=item_key, rating_key=rating_key)
        cv_split.append((train, test))

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i + 1, space_size, params))
        tried_conf.append(params)
        cv_result = 0.0
        for f, (train, test) in enumerate(cv_split):
            # train the recommender
            recommender = RecommenderClass(**params)
            recommender.fit(train)
            # evaluate the ranking quality
            n_eval = 0
            metric_ = 0.0
            for test_user in range(nusers):
                relevant_items = test[test_user].indices
                if len(relevant_items) > 0:
                    n_eval += 1
                    # this will rank **all** items
                    recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True)
                    # evaluate the recommendation list with ranking metrics ONLY
                    if metric == roc_auc:
                        metric_ += roc_auc(recommended_items, relevant_items)
                    elif metric == ndcg:
                        metric_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
                    else:
                        metric_ += metric(recommended_items, relevant_items, at=at)
            metric_ /= n_eval
            cv_result += metric_
        # average value of the metric in cross-validation
        results[i] = cv_result / cv_folds
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]


def random_search_cv(RecommenderClass, dataset, param_space, iters=10, metric=roc_auc, at=None,
                     cv_folds=5, is_implicit=True, user_key='user_id', item_key='item_id', rating_key='rating',
                     rnd_seed=1234):
    """
    Finds the best hyper-parameters of a recommender algorithm with Random Search

    :param RecommenderClass: Class of the recommender to tune (must be subclass of Recommender)
    :param dataset: data to use for tuning
    :param param_space: space of the parameters to explore
    :param iters: number of iterations of Random Search
    :param metric: metric to maximize
    :param at: optional length of the recommendation list used in recommendaiton
    :param cv_folds: number of cross-validation iters
    :param is_implicit: True to discard ratings, False otherwise
    :param user_key: name of the column with user ids in dataset
    :param item_key: name of the column with item ids in dataset
    :param rating_key: name of the column with ratings in dataset
    :param rnd_seed: random seed used for cross-validation
    :return: a tuple with (best configuration, best metric value)
    """

    tried_conf = []
    space_size = iters
    # sample `iter` configurations at random from param_space
    param_list = list(ParameterSampler(param_space, n_iter=iters, random_state=rnd_seed))
    results = np.zeros((space_size,), dtype=np.float32)
    logger.info('Size of the parameter space: {} ({} cv trials)'.format(space_size, space_size * cv_folds))
    # compute the cv splits
    nusers, nitems = dataset[user_key].max() + 1, dataset[item_key].max() + 1
    cv_split = []
    for train_df, test_df in k_fold_cv(dataset,
                                       user_key=user_key,
                                       item_key=item_key,
                                       k=cv_folds,
                                       clean_test=True,
                                       seed=rnd_seed):
        train = df_to_csr(train_df, is_implicit=is_implicit, nrows=nusers, ncols=nitems,
                          user_key=user_key, item_key=item_key, rating_key=rating_key)
        test = df_to_csr(test_df, is_implicit=is_implicit, nrows=nusers, ncols=nitems,
                         user_key=user_key, item_key=item_key, rating_key=rating_key)
        cv_split.append((train, test))

    for i, params in enumerate(param_list):
        logger.info('Iteration {}/{}: {}'.format(i + 1, space_size, params))
        tried_conf.append(params)
        cv_result = 0.0
        for f, (train, test) in enumerate(cv_split):
            # train the recommender
            recommender = RecommenderClass(**params)
            recommender.fit(train)
            # evaluate the ranking quality
            n_eval = 0
            metric_ = 0.0
            for test_user in range(nusers):
                relevant_items = test[test_user].indices
                if len(relevant_items) > 0:
                    n_eval += 1
                    # this will rank **all** items
                    recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True)
                    # evaluate the recommendation list with ranking metrics ONLY
                    if metric == roc_auc:
                        metric_ += roc_auc(recommended_items, relevant_items)
                    elif metric == ndcg:
                        metric_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
                    else:
                        metric_ += metric(recommended_items, relevant_items, at=at)
            metric_ /= n_eval
            cv_result += metric_
        # average value of the metric in cross-validation
        results[i] = cv_result / cv_folds
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]
