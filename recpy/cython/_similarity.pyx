# cython: profile=True
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

@cython.boundscheck(False)
def cosine_common(X):
    """
    Function that pairwise cosine similarity of the columns in X.
    It takes only the values in common between each pair of columns
    :param X: instance of scipy.sparse.csc_matrix
    :return:
        the result of co_prodsum
        the number of co_rated elements for every column pair
    """
    if not isinstance(X, sps.csc_matrix):
        raise ValueError('X must be an instance of scipy.sparse.csc_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of X
    cdef int [:] indices = X.indices, indptr = X.indptr
    cdef float [:] data = X.data

    # initialize the result variables
    cdef int ncols = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros([ncols, ncols], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] common = np.zeros([ncols, ncols], dtype=np.int32)

    cdef int i, j, n_i, n_j, ii, jj, n_common
    cdef float ii_sum, jj_sum, ij_sum, x_i, x_j

    # TODO: compute the cosine similarity over the COMMON items of each pair of columns
    # we will also return the number of common items between each pair of columns
    return result, common