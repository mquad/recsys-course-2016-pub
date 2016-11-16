import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from sklearn.linear_model import ElasticNet


class SLIM(Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model ElasticNet linear_model.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        super(SLIM, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def fit(self, X):
        self.dataset = X
        # TODO: build the SLIM W matrix
        pass

    def recommend(self, user_id, n=None, exclude_seen=True):
        # TODO: compute the scores using the dot product
        pass

from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM(SLIM):
    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 workers=4):
        super(MultiThreadSLIM, self).__init__(l1_penalty=l1_penalty,
                                              l2_penalty=l2_penalty,
                                              positive_only=positive_only)
        self.workers = workers

    def __str__(self):
        return "SLIM_mt (l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )


    def fit(self, X):
        self.dataset = X
        # TODO: build the SLIM W matrix in a parallelized way
        pass

