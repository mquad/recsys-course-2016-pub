# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-10-24 15:27:46
# @Last Modified by:   massimo
# @Last Modified time: 2016-10-24 17:03:51
import numpy as np
import scipy.sparse as sps


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


class Recommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(Recommender, self).__init__()
        self.dataset = None

    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]

    def fit(self, X):
        pass

    def recommend(self, user_id, n=None, exclude_seen=True):
        pass

    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]
