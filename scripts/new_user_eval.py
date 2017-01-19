import argparse
import logging
import pandas as pd
from collections import OrderedDict
from datetime import datetime as dt

from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.utils.split import split_by_user, per_user_holdout
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr

from recpy.recommenders.item_knn import ItemKNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.non_personalized import TopPop

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('item_knn', ItemKNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--is_binary', action='store_true', default=False)
parser.add_argument('--make_binary', action='store_true', default=False)
parser.add_argument('--binary_th', type=float, default=4.0)
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--n_observed', type=int, default=1)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender', type=str, default='top_pop')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=10)
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_recommenders, 'Unknown recommender: {}'.format(args.recommender)
RecommenderClass = available_recommenders[args.recommender]
# parse recommender parameters
init_args = OrderedDict()
if args.params:
    for p_str in args.params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

# convert the column argument to list
if args.columns is not None:
    args.columns = args.columns.split(',')

# read the dataset
logger.info('Reading {}'.format(args.dataset))
dataset, item_to_idx, user_to_idx = read_dataset(
    args.dataset,
    header=args.header,
    sep=args.sep,
    columns=args.columns,
    make_binary=args.make_binary,
    binary_th=args.binary_th,
    item_key=args.item_key,
    user_key=args.user_key,
    rating_key=args.rating_key)

nusers, nitems = dataset.user_idx.max() + 1, dataset.item_idx.max() + 1
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# compute the user holdout split
logger.info('Computing the {:.0f}% user holdout split'.format(args.holdout_perc * 100))
train_users_df, test_users_df, train_user_to_idx, test_user_to_idx = split_by_user(dataset,
                                                                                   user_key=args.user_key,
                                                                                   item_key=args.item_key,
                                                                                   perc=args.holdout_perc,
                                                                                   seed=args.rnd_seed,
                                                                                   compress_user_indices=True)
nusers_train = train_users_df.user_idx.max() + 1
nusers_test = test_users_df.user_idx.max() + 1
logger.info('Train shape: ({},{})'.format(nusers_train, nitems))
logger.info('Test shape: ({},{})'.format(nusers_test, nitems))

# the split the ratings for cold users into observed and hidden portions
logger.info('Partitioning new user\'s activity into observed and hidden ratings (n_observed={})'.format(
    args.n_observed))
test_observed_df, test_hidden_df = per_user_holdout(test_users_df,
                                                    user_key='user_idx',
                                                    item_key='item_idx',
                                                    n_observed=args.n_observed,
                                                    seed=args.rnd_seed)
tot_observed, tot_hidden = test_observed_df.shape[0], test_hidden_df.shape[0]
logger.info('Observed ratings: {}({:.2f}%)'.format(tot_observed, tot_observed/(tot_observed+tot_hidden)*100))
logger.info('Observed ratings: {}({:.2f}%)'.format(tot_hidden, tot_hidden/(tot_observed+tot_hidden)*100))

# build the sparse matrices
train = df_to_csr(train_users_df,
                  is_binary=args.is_binary,
                  nrows=nusers_train,
                  ncols=nitems,
                  item_key='item_idx',
                  user_key='user_idx',
                  rating_key=args.rating_key)
test_observed = df_to_csr(test_observed_df,
                          is_binary=args.is_binary,
                          nrows=nusers_test,
                          ncols=nitems,
                          item_key='item_idx',
                          user_key='user_idx',
                          rating_key=args.rating_key)

test_hidden = df_to_csr(test_hidden_df,
                        is_binary=args.is_binary,
                        nrows=nusers_test,
                        ncols=nitems,
                        item_key='item_idx',
                        user_key='user_idx',
                        rating_key=args.rating_key)

# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
tic = dt.now()
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed in {}'.format(dt.now() - tic))

# evaluate the ranking quality
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
at = args.rec_length
n_eval = 0
for test_user in range(nusers_test):
    user_profile = test_observed[test_user]
    relevant_items = test_hidden[test_user].indices
    if len(relevant_items) > 0:
        n_eval += 1
        # this will rank **all** items
        recommended_items = recommender.recommend_new_user(user_profile=user_profile, exclude_seen=False)

        # evaluate the recommendation list with ranking metrics ONLY
        roc_auc_ += roc_auc(recommended_items, relevant_items)
        precision_ += precision(recommended_items, relevant_items, at=at)
        recall_ += recall(recommended_items, relevant_items, at=at)
        map_ += map(recommended_items, relevant_items, at=at)
        mrr_ += rr(recommended_items, relevant_items, at=at)
        ndcg_ += ndcg(recommended_items, relevant_items, relevance=test_hidden[test_user].data, at=at)
roc_auc_ /= n_eval
precision_ /= n_eval
recall_ /= n_eval
map_ /= n_eval
mrr_ /= n_eval
ndcg_ /= n_eval

logger.info('Ranking quality')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_))
logger.info('Precision@{}: {:.4f}'.format(at, precision_))
logger.info('Recall@{}: {:.4f}'.format(at, recall_))
logger.info('MAP@{}: {:.4f}'.format(at, map_))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_))
