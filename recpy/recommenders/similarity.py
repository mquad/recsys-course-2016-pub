import numpy as np
import scipy.sparse as sps
from .base import check_matrix
from ..cython._similarity import cosine_common


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # TODO: compute the cosine similarity for each pair of columns in X
        pass

    def apply_shrinkage(self, X, sim):
        # TODO: compute the shrunk similarity
        return sim

class Pearson(ISimilarity):
    def compute(self, X):
        # TODO: compute the Pearson similarity for each pair of columns in X
        pass


class AdjustedCosine(ISimilarity):
    def compute(self, X):
        # TODO: compute the Adjusted Cosine similarity for each pair of columns in X
        pass
