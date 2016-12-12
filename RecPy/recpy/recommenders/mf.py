import numpy as np
from .base import Recommender, check_matrix
from .._cython._mf import FunkSVD_sgd, AsySVD_sgd, AsySVD_compute_user_factors
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")


class FunkSVD(Recommender):
    # TODO: add global effects
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        super(FunkSVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "FunkSVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, X):
        self.dataset = X
        X = check_matrix(X, 'csr', dtype=np.float32)
        # TODO: complete the Cython function FunkSVD_sgd
        # that returns the matrices of factors U and V 
        self.U, self.V = FunkSVD_sgd(X, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)


    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.U[user_id], self.V.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]


class AsySVD(Recommender):
    # TODO: add global effects
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        super(AsySVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "AsySVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        # TODO: complete the Cython function AsySVD_sgd
        # that returns the matrices of factors X and Y
        self.X, self.Y = AsySVD_sgd(R, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                    self.init_std,
                                    self.lrate_decay, self.rnd_seed)
        # precompute the user factors
        M = R.shape[0]
        self.U = np.vstack([AsySVD_compute_user_factors(R[i], self.Y) for i in range(M)])

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X, self.U[user_id].T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]


class IALS_numpy(Recommender):
    # TODO: Add support for multiple confidence scaling functions
    def __init__(self,
                 num_factors=50,
                 reg=0.015,
                 iters=10,
                 alpha=40,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        super(IALS_numpy, self).__init__()
        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.alpha = alpha
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "WRMF-iALS(num_factors={},  reg={}, iters={}, alpha={}, init_mean={}, " \
               "init_std={}, rnd_seed={})".format(
            self.num_factors, self.reg, self.iters, self.alpha, self.init_mean, self.init_std,
            self.rnd_seed
        )

    def fit(self, X):
        self.dataset = X
        #
        # TODO: learn the U and V factors with Alternating Least Squares
        # This time, let's use python and numpy only
        #
        self.U, self.V = None, None

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.U[user_id], self.V.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]