import argparse
import logging
from collections import OrderedDict
from datetime import datetime as dt

from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.utils.split import k_fold_cv
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr

from recpy.recommenders.item_knn import ItemKNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.mf import FunkSVD, IALS_numpy, AsySVD
from recpy.recommenders.non_personalized import TopPop, GlobalEffects

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('global_effects', GlobalEffects),
    ('item_knn', ItemKNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
    ('FunkSVD', FunkSVD),
    ('AsySVD', AsySVD),
    ('IALS_np', IALS_numpy),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--is_implicit', action='store_true', default=False)
parser.add_argument('--make_implicit', action='store_true', default=False)
parser.add_argument('--implicit_th', type=float, default=4.0)
parser.add_argument('--cv_folds', type=int, default=5)
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
dataset, idx_to_user, idx_to_item = read_dataset(
    args.dataset,
    header=args.header,
    sep=args.sep,
    columns=args.columns,
    make_implicit=args.make_implicit,
    implicit_th=args.implicit_th,
    item_key=args.item_key,
    user_key=args.user_key,
    rating_key=args.rating_key)

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# evaluate the recommendation quality with k-fold cross-validation
logger.info('Running {}-fold Cross Validation'.format(args.cv_folds))
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = np.zeros(args.cv_folds), np.zeros(args.cv_folds), np.zeros(
    args.cv_folds), np.zeros(args.cv_folds), np.zeros(args.cv_folds), np.zeros(args.cv_folds)
at = args.rec_length
nfold = 0
for train_df, test_df in k_fold_cv(dataset,
                                   user_key=args.user_key,
                                   item_key=args.item_key,
                                   k=args.cv_folds,
                                   clean_test=True,
                                   seed=args.rnd_seed):
    logger.info(train_df.shape)
    logger.info(test_df.shape)
    logger.info('Fold {}'.format(nfold + 1))
    train = df_to_csr(train_df, is_implicit=args.is_implicit, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, is_implicit=args.is_implicit, nrows=nusers, ncols=nitems)

    # train the recommender
    recommender = RecommenderClass(**init_args)
    logger.info('Recommender: {}'.format(recommender))
    tic = dt.now()
    logger.info('Training started')
    recommender.fit(train)
    logger.info('Training completed in {}'.format(dt.now() - tic))

    # evaluate the ranking quality
    n_eval = 0
    for test_user in range(nusers):
        user_profile = train[test_user]
        relevant_items = test[test_user].indices
        if len(relevant_items) > 0:
            n_eval += 1
            # this will rank **all** items
            recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True)

            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_[nfold] += roc_auc(recommended_items, relevant_items)
            precision_[nfold] += precision(recommended_items, relevant_items, at=at)
            recall_[nfold] += recall(recommended_items, relevant_items, at=at)
            map_[nfold] += map(recommended_items, relevant_items, at=at)
            mrr_[nfold] += rr(recommended_items, relevant_items, at=at)
            ndcg_[nfold] += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
    roc_auc_[nfold] /= n_eval
    precision_[nfold] /= n_eval
    recall_[nfold] /= n_eval
    map_[nfold] /= n_eval
    mrr_[nfold] /= n_eval
    ndcg_[nfold] /= n_eval

    nfold += 1

logger.info('Ranking quality')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_.mean()))
logger.info('Precision@{}: {:.4f}'.format(at, precision_.mean()))
logger.info('Recall@{}: {:.4f}'.format(at, recall_.mean()))
logger.info('MAP@{}: {:.4f}'.format(at, map_.mean()))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_.mean()))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_.mean()))
