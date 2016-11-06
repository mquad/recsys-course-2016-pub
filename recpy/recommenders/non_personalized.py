import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix


class TopPop(Recommender):
    """Top Popular recommender"""

    def __init__(self):
        super(TopPop, self).__init__()

    def fit(self, X):
        self.dataset = X
        # convert to csc matrix for faster column-wise sum
        X = check_matrix(X, 'csc', dtype=np.float32)
        item_pop = (X > 0).sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)
        item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
        self.pop = np.argsort(item_pop)[::-1]

    def recommend(self, user_id, n=None, exclude_seen=True):
        ranking = self.pop
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def __str__(self):
        return "TopPop"


class GlobalEffects(Recommender):
    """docstring for GlobalEffects"""

    def __init__(self, lambda_user=10, lambda_item=25):
        super(GlobalEffects, self).__init__()
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item

    def fit(self, X):
        self.dataset = X
        # convert to csc matrix for faster column-wise sum
        X = check_matrix(X, 'csc', dtype=np.float32)
        # 1) global average
        self.mu = X.data.sum(dtype=np.float32) / X.data.shape[0]

        # 2) item average bias
        # compute the number of non-zero elements for each column
        col_nnz = np.diff(X.indptr)

        # it is equivalent to:
        # col_nnz = X.indptr[1:] - X.indptr[:-1]
        # and it is **much faster** than
        # col_nnz = (X != 0).sum(axis=0)

        X_unbiased = X.copy()
        X_unbiased.data -= self.mu
        self.bi = X_unbiased.sum(axis=0) / (col_nnz + self.lambda_item)
        self.bi = np.asarray(self.bi).ravel()  # converts 2-d matrix to 1-d array without anycopy

        # 3) user average bias
        # NOTE: the user bias is *useless* for the sake of ranking items. We just show it here for educational purposes.

        # first subtract the item biases from each column
        # then repeat each element of the item bias vector a number of times equal to col_nnz
        # and subtract it from the data vector
        X_unbiased.data -= np.repeat(self.bi, col_nnz)

        # now convert the csc matrix to csr for efficient row-wise computation
        X_csr = X_unbiased.tocsr()
        row_nnz = np.diff(X_csr.indptr)
        # finally, let's compute the bias
        self.bu = X_csr.sum(axis=1).ravel() / (row_nnz + self.lambda_user)

        # 4) precompute the item ranking by using the item bias only
        # the global average and user bias won't change the ranking, so there is no need to use them
        self.item_ranking = np.argsort(self.bi)[::-1]

    def recommend(self, user_id, k=None, exclude_seen=True):
        ranking = self.item_ranking
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:k]

    def __str__(self):
        return 'GlobalEffects'
