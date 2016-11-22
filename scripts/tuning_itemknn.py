import logging
import argparse
import pandas as pd
import numpy as np

from recpy.recommenders import ItemKNNRecommender
from recpy.recommenders import TopPop, GlobalEffects
from recpy.recommenders import SLIM, MultiThreadSLIM
from recpy.utils.data_utils import read_dataset, df_to_csr
from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

train_df = read_dataset('data/ml100k/train_tuning.csv', sep=',', header=0)
test_df = read_dataset('data/ml100k/valid.csv', sep=',', header=0)

nusers, nitems = train_df.user_idx.max()+1, train_df.item_idx.max()+1
train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

shrinkage = np.arange(0, 200, 25)
result = np.zeros_like(shrinkage, dtype=np.float32)
metric = recall
at = 10

for idx, sh in enumerate(shrinkage):
    logger.info("Iter {}/{}".format(idx+1, len(shrinkage)))
    recommender = ItemKNNRecommender(shrinkage=sh, similarity='cosine')
    recommender.fit(train)
    metric_ = 0.0
    n_eval = 0
    for test_user in range(nusers):
        user_profile = train[test_user]
        relevant_items = test[test_user].indices
        if len(relevant_items) > 0:
            n_eval += 1
            recommended_items = recommender.recommend(user_id=test_user, exclude_seen=True, n=at)

            # evaluate the recommendation list with ranking metrics ONLY
            metric_ += metric(recommended_items, relevant_items, at=at)
    metric_ /= n_eval
    result[idx] = metric_

logger.info(shrinkage)
logger.info(result)