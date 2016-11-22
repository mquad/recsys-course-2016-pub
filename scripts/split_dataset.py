import argparse
import logging

from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.utils.split import holdout

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('train_file')
parser.add_argument('test_file')
parser.add_argument('--is_implicit', action='store_true', default=False)
parser.add_argument('--make_implicit', action='store_true', default=False)
parser.add_argument('--implicit_th', type=float, default=4.0)
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
    make_implicit=args.make_implicit,
    implicit_th=args.implicit_th,
    item_key=args.item_key,
    user_key=args.user_key,
    rating_key=args.rating_key)

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# compute the holdout split
logger.info('Computing the {:.0f}% holdout split'.format(args.holdout_perc * 100))
train_df, test_df = holdout(dataset,
                            user_key=args.user_key,
                            item_key=args.item_key,
                            perc=args.holdout_perc,
                            clean_test=True,
                            seed=args.rnd_seed)
logger.info('Writing the training split to {}'.format(args.train_file))
train_df.to_csv(args.train_file,
                index=False,
                sep=args.sep,
                header=args.header is not None)
logger.info('Writing the test split to {}'.format(args.test_file))
test_df.to_csv(args.test_file,
                index=False,
                sep=args.sep,
                header=args.header is not None)
logger.info('Done')