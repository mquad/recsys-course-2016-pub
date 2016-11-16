import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine


class ItemKNNRecommender(Recommender):
    """ ItemKNN recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        super(ItemKNNRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = similarity
        self.sparse_weights = sparse_weights
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "ItemKNN(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        self.dataset = X
        item_weights = self.distance.compute(X)
        # for each column, keep only the top-k scored items
        idx_sorted = np.argsort(item_weights, axis=0)  # sort by column
        if not self.sparse_weights:
            self.W = item_weights.copy()
            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-self.k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            self.W[not_top_k, np.arange(item_weights.shape[1])] = 0.0
        else:
            # iterate over each column and keep only the top-k similar items
            values, rows, cols = [], [], []
            nitems = self.dataset.shape[1]
            for i in range(nitems):
                top_k_idx = idx_sorted[-self.k:, i]
                values.extend(item_weights[top_k_idx, i])
                rows.extend(np.arange(nitems)[top_k_idx])
                cols.extend(np.ones(self.k) * i)
            self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        if self.sparse_weights:
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            scores = user_profile.dot(self.W).ravel()
        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            den = rated.dot(self.item_weights).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]
