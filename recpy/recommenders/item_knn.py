import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine


class ItemKNNRecommender(Recommender):
    """ ItemKNN recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False):
        super(ItemKNNRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.item_weights = None
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = similarity
        if similarity == 'cosine':
            self.similarity = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.similarity = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.similarity = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "ItemKNN(similarity={})".format(self.similarity_name)

    def fit(self, X):
        self.dataset = X
        self.item_weights = self.similarity.compute(X)
        # for each column, keep only the top-k scored items
        idx_sorted = np.argsort(self.item_weights, axis=0) # sort by column
        # index of the items that DON'T BELONG 
        # to the top-k similar items
        not_top_k = idx_sorted[:-self.k, :]
        # zero-out the not top-k items for each column
        self.item_weights[not_top_k, np.arange(self.item_weights.shape[1])] = 0.0


    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.item_weights).ravel()

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(user_profile.data)
            den = rated.dot(self.item_weights).ravel()
            den[np.abs(den) < 1e-6] = 1.0 # to avoid NaNs
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]