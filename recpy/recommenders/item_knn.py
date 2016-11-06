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
        # TODO: for each column, keep only the top-k scored items

    def recommend(self, user_id, n=None, exclude_seen=True):
        # TODO: compute the scores
        if self.normalize:
            # TODO: normalize the scores
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            pass
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]