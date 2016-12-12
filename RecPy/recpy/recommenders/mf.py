import numpy as np
from .base import Recommender, check_matrix
from .._cython._mf import FunkSVD_sgd, AsySVD_sgd, AsySVD_compute_user_factors
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")


class FunkSVD(Recommender):
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    '''
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
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
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
    '''
    AsymmetricSVD model
    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    '''
    # TODO: add global effects
    # TODO: recommendation for new-users. Update the precomputed profiles online
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
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
    '''
    Implicit Alternating Least Squares model (or Weighed Regularized Matrix Factorization)
    Reference: Collaborative Filtering for Implicit Feedback Datasets (Hu et al., 2008)

    Factorization model for implicit feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    '''

    # TODO: Add support for multiple confidence scaling functions (e.g. linear and log scaling)
    def __init__(self,
                 num_factors=50,
                 reg=0.015,
                 iters=10,
                 alpha=40,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param alpha: scaling factor to compute confidence scores
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

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

    def fit(self, R):
        self.dataset = R
        # compute the confidence matrix
        C = R.copy().tocsr()
        # use linear scaling here
        # TODO: add log-scaling
        C.data = 1 + self.alpha * C.data
        Ct = C.T.tocsr()
        M, N = R.shape

        # set the seed
        np.random.seed(self.rnd_seed)

        # initialize the latent factors
        self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
        self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

        for it in range(self.iters):
            self.X = self._lsq_solver(C, self.X, self.Y, self.reg)
            self.Y = self._lsq_solver(Ct, self.Y, self.X, self.reg)
            logger.debug('Finished iter {}'.format(it + 1))

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def _lsq_solver(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            # accumulate Yt*Ci*p(i) in b
            b = np.zeros(factors)

            for j, cij in self._nonzeros(C, i):
                vj = Y[j]
                A += (cij - 1.0) * np.outer(vj, vj)
                b += cij * vj

            X[i] = np.linalg.solve(A, b)
        return X

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X

    def _nonzeros(self, R, row):
        for i in range(R.indptr[row], R.indptr[row + 1]):
            yield (R.indices[i], R.data[i])
