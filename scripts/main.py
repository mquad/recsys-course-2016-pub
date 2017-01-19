import argparse
import logging
import pandas as pd
from collections import OrderedDict
from datetime import datetime as dt

from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr

from recpy.recommenders.item_knn import ItemKNNRecommender
from recpy.recommenders.user_knn import UserKNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.mf import FunkSVD, IALS_numpy, AsySVD, BPRMF
from recpy.recommenders.non_personalized import TopPop, GlobalEffects

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('global_effects', GlobalEffects),
    ('item_knn', ItemKNNRecommender),
    ('user_knn', UserKNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
    ('FunkSVD', FunkSVD),
    ('AsySVD', AsySVD),
    ('IALS_np', IALS_numpy),
    ('BPRMF', BPRMF),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('train')
parser.add_argument('test')
parser.add_argument('--is_binary', action='store_true', default=False)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender', type=str, default='top_pop')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--prediction_file', type=str, default=None)
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
logger.info('Reading {}'.format(args.train))
train_df, item_to_idx, user_to_idx = read_dataset(args.train,
                                                  sep=',',
                                                  header=0,
                                                  columns=args.columns,
                                                  user_key=args.user_key,
                                                  item_key=args.item_key,
                                                  rating_key=args.rating_key)
logger.info('Reading {}'.format(args.test))
test_df, _, _ = read_dataset(args.test,
                             sep=',',
                             header=0,
                             columns=args.columns,
                             user_key=args.user_key,
                             item_key=args.item_key,
                             rating_key=args.rating_key,
                             item_to_idx=item_to_idx,
                             user_to_idx=user_to_idx)

# build reverse maps
idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)

nusers, nitems = train_df.user_idx.max() + 1, train_df.item_idx.max() + 1
train = df_to_csr(train_df,
                  is_binary=args.is_binary,
                  nrows=nusers,
                  ncols=nitems,
                  user_key='user_idx',
                  item_key='item_idx',
                  rating_key=args.rating_key)
test = df_to_csr(test_df,
                 is_binary=args.is_binary,
                 nrows=nusers,
                 ncols=nitems,
                 user_key='user_idx',
                 item_key='item_idx',
                 rating_key=args.rating_key)

# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
tic = dt.now()
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed in {}'.format(dt.now() - tic))

# open the prediction file
if args.prediction_file:
    pfile = open(args.prediction_file, 'w')
    n = args.rec_length if args.rec_length is not None else nitems
    header = 'user_id,'
    header += ','.join(['rec_item{}'.format(i + 1) for i in range(args.rec_length)]) + '\n'
    pfile.write(header)

# evaluate the ranking quality
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
at = args.rec_length
n_eval = 0

for test_user in range(nusers):
    user_profile = train[test_user]
    relevant_items = test[test_user].indices
    if len(relevant_items) > 0:
        n_eval += 1
        # this will rank **all** items
        recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True)

        if args.prediction_file:
            # write the recommendation list to file, one user per line
            # TODO: convert user and item indices back to their original ids
            user_id = test_user
            rec_list = recommended_items[:args.rec_length]
            s = str(user_id) + ','
            s += ','.join([str(x) for x in rec_list]) + '\n'
            pfile.write(s)

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

# close the prediction file
if args.prediction_file:
    pfile.close()
    logger.info('Recommendations written to {}'.format(args.prediction_file))

logger.info('Ranking quality')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_))
logger.info('Precision@{}: {:.4f}'.format(at, precision_))
logger.info('Recall@{}: {:.4f}'.format(at, recall_))
logger.info('MAP@{}: {:.4f}'.format(at, map_))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_))
