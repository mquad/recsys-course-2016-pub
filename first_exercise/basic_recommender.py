import numpy as np
import scipy.sparse as sps
import pandas as pd
import logging
import argparse
from metrics import roc_auc, precision, recall, map, ndcg, rr
from datetime import datetime as dt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def read_dataset(path, header=None, columns=None, user_key='user_id', item_key='item_id', rating_key='rating', sep=','):
	data = pd.read_csv(path, header=header, names=columns, sep=sep)
	logger.info('Columns: {}'.format(data.columns.values))
	# build user and item maps (and reverse maps)
	# this is used to map ids to indexes starting from 0 to nitems (or nusers)
	items = data[item_key].unique()
	users = data[user_key].unique()
	item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
	user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
	idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
	idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
	# map ids to indices
	data['item_idx'] = item_to_idx[data[item_key].values].values
	data['user_idx'] = user_to_idx[data[user_key].values].values
	return data, idx_to_user, idx_to_item

def holdout_split(data, perc=0.8, seed=1234):
	# set the random seed
	rng = np.random.RandomState(seed)
	# shuffle data
	nratings = data.shape[0]
	shuffle_idx = rng.permutation(nratings)
	train_size = int(nratings * perc)
	# split data according to the shuffled index and the holdout size
	train_split = data.ix[shuffle_idx[:train_size]]
	test_split = data.ix[shuffle_idx[train_size:]]
	return train_split, test_split

def df_to_csr(df, nrows, ncols, user_key='user_idx', item_key='item_idx', rating_key='rating'):
	rows = df[user_key].values
	columns = df[item_key].values
	ratings = df[rating_key].values
	shape = (nrows, ncols)
	# using the 4th constructor of csr_matrix 
	# reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
	return sps.csr_matrix((ratings, (rows, columns)), shape=shape)

class TopPop(object):
	"""Top Popular recommender"""
	def __init__(self):
		super(TopPop, self).__init__()
	def fit(self, train):
		if not isinstance(train, sps.csc_matrix):
			# convert to csc matrix for faster column-wise sum
			train_csc = train.tocsc()
		else:
			train_csc = train
		item_pop = (train_csc > 0).sum(axis=0)	# this command returns a numpy.matrix of size (1, nitems)
		item_pop = np.asarray(item_pop).squeeze() # necessary to convert it into a numpy.array of size (nitems,)
		self.pop = np.argsort(item_pop)[::-1]
	def recommend(self, profile, k=None, exclude_seen=True):
		unseen_mask = np.in1d(self.pop, profile, assume_unique=True, invert=True)
		return self.pop[unseen_mask][:k]
		
# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
args = parser.parse_args()

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
	item_key=args.item_key,
	user_key=args.user_key,
	rating_key=args.rating_key)

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# compute the holdout split
logger.info('Computing the {:.0f}% holdout split'.format(args.holdout_perc*100))
train_df, test_df = holdout_split(dataset, perc=args.holdout_perc, seed=args.rnd_seed)
train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

# top-popular recommender
logger.info('Building the top-popular recommender')
recommender = TopPop()
tic = dt.now()
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed built in {}'.format(dt.now() - tic))

# ranking quality evaluation
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
at = 20
neval = 0
for test_user in range(nusers):
	user_profile = train[test_user].indices #what is doing here?
	relevant_items = test[test_user].indices
	if len(relevant_items) > 0:
		neval += 1
		#
		# TODO: Here you can write to file the recommendations for each user in the test split. 
		# WARNING: there is a catch with the item idx!
		#
		# this will rank *all* items
		recommended_items = recommender.recommend(user_profile, exclude_seen=True)	
		# use this to have the *top-k* recommended items (warning: this can underestimate ROC-AUC for small k)
		# recommended_items = recommender.recommend(user_profile, k=at, exclude_seen=True)	
		roc_auc_ += roc_auc(recommended_items, relevant_items)
		precision_ += precision(recommended_items, relevant_items, at=at)
		recall_ += recall(recommended_items, relevant_items, at=at)
		map_ += map(recommended_items, relevant_items, at=at)
		mrr_ += rr(recommended_items, relevant_items, at=at)
		ndcg_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
roc_auc_ /= neval
precision_ /= neval
recall_ /= neval
map_ /= neval
mrr_ /= neval
ndcg_ /= neval

logger.info('Ranking quality')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_))
logger.info('Precision@{}: {:.4f}'.format(at, precision_))
logger.info('Recall@{}: {:.4f}'.format(at, recall_))
logger.info('MAP@{}: {:.4f}'.format(at, map_))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_))
