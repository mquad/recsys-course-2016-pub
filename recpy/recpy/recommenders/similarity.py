import numpy as np
import scipy.sparse as sps
from .base import check_matrix
from .._cython._similarity import cosine_common


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)

        # 2) compute the cosine similarity using the dot-product
        dist = X.T.dot(X).toarray()
        # zero out diagonal values
        np.fill_diagonal(dist, 0.0)
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind.T.dot(X_ind).toarray().astype(np.float32)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        dist *= co_counts / (co_counts + self.shrinkage)
        return dist


class Pearson(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)
        # subtract the item average rating
        col_nnz = np.diff(X.indptr)
        col_means = np.asarray(X.sum(axis=0) / (col_nnz + 1e-6)).ravel()
        X.data -= np.repeat(col_means, col_nnz)

        dist, co_counts = cosine_common(X)
        if self.shrinkage > 0:
            dist *= co_counts / (co_counts + self.shrinkage)
        return dist


class AdjustedCosine(ISimilarity):
    def compute(self, X):
        # convert X to csr matrix for faster row-wise operations
        X = check_matrix(X, 'csr', dtype=np.float32)
        # subtract the user average rating
        row_nnz = np.diff(X.indptr)
        row_means = np.asarray(X.sum(axis=1).ravel() / (row_nnz + 1e-6)).ravel()
        X.data -= np.repeat(row_means, row_nnz)

        # convert X to csc before applying cosine_common
        X = X.tocsc()
        dist, co_counts = cosine_common(X)
        if self.shrinkage > 0:
            dist *= co_counts / (co_counts + self.shrinkage)
        return dist


# from .._cython._similarity import pearson_corr
# class Pearson2(IDistance):
#     def compute(self, X):
#         # convert to csc matrix for faster column-wise operations
#         X = check_matrix(X, 'csc', dtype=np.float32)
#         dist, co_counts = pearson_corr(X)
#         if self.shrinkage > 0:
#             dist *= co_counts / (co_counts + self.shrinkage)
#         return dist

