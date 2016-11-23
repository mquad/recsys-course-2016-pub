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

    # let's declare all the variables that we'll use in the loop here
    # NOTE: declaring the type of your variables makes your Cython code run MUCH faster
    # NOTE: Cython allows cdef's only in the main scope
    # cdef's in nested codes will result in compilation errors
    cdef int i, j, n_i, n_j, ii, jj, n_common
    cdef float ii_sum, jj_sum, ij_sum, x_i, x_j

    for i in range(ncols):
        n_i = indptr[i+1] - indptr[i]
        # the correlation matrix is symmetric,
        # let's compute only the values for the upper-right triangle
        for j in range(i+1, ncols):
            n_j = indptr[j+1] - indptr[j]

            ij_sum, ii_sum, jj_sum = 0.0, 0.0, 0.0
            ii, jj = 0, 0
            n_common = 0

            # here we exploit the fact that the two subvectors in indices are sorted
            # to compute the dot product of the rows in common between i and j in linear time.
            # (indices[indptr[i]:indptr[i]+n_i] and indices[indptr[j]:indptr[j]+n_j]
            # contain the row indices of the non-zero items in columns i and j)
            while ii < n_i and jj < n_j:
                if indices[indptr[i] + ii] < indices[indptr[j] + jj]:
                    ii += 1
                elif indices[indptr[i] + ii] > indices[indptr[j] + jj]:
                    jj += 1
                else:
                    x_i = data[indptr[i] + ii]
                    x_j = data[indptr[j] + jj]
                    ij_sum += x_i * x_j
                    ii_sum += x_i ** 2
                    jj_sum += x_j ** 2
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                result[i, j] = ij_sum / np.sqrt(ii_sum * jj_sum)
                result[j, i] = result[i, j]
                common[i, j] = n_common
                common[j, i] = n_common
    return result, common


@cython.boundscheck(False)
def pearson_corr(X):
    """
    Pearson correlation
    :param X: instance of scipy.sparse.csc_matrix
    :return:
        the pairwise Pearson correlation matrix
        the number of co_rated elements for every column pair
    """
    if not isinstance(X, sps.csc_matrix):
        raise ValueError('X must be an instance of scipy.sparse.csc_matrix')

    # use MemoryViews for fast access to the sparse structure of X
    cdef int [:] indices = X.indices, indptr = X.indptr
    cdef float [:] data = X.data

    # initialize the result variables
    cdef int ncols = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros([ncols, ncols], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] common = np.zeros([ncols, ncols], dtype=np.int32)

    cdef int i, j, n_i, n_j, ii, jj, n_common
    cdef float i_sum, j_sum, ii_sum, jj_sum, ij_sum, x_i, x_j

    for i in range(ncols):
        n_i = indptr[i+1] - indptr[i]
        # the correlation matrix is symmetric,
        # let's compute only the values for the upper-right triangle
        for j in range(i, ncols):
            n_j = indptr[j+1] - indptr[j]

            i_sum, j_sum = 0.0, 0.0
            ij_sum, ii_sum, jj_sum = 0.0, 0.0, 0.0
            ii, jj = 0, 0
            n_common = 0

            while ii < n_i and jj < n_j:
                if indices[indptr[i] + ii] < indices[indptr[j] + jj]:
                    ii += 1
                elif indices[indptr[i] + ii] > indices[indptr[j] + jj]:
                    jj += 1
                else:
                    x_i = data[indptr[i] + ii]
                    x_j = data[indptr[j] + jj]
                    ij_sum += x_i * x_j
                    i_sum += x_i
                    j_sum += x_j
                    ii_sum += x_i * x_i
                    jj_sum += x_j * x_j
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                num = n_common * ij_sum - i_sum * j_sum
                den = np.sqrt((n_common * ii_sum - i_sum * i_sum) * (n_common * jj_sum - j_sum * j_sum))
                if den > 0.0:
                    c = num / den
                else:
                    c = 0.0
                result[i, j] = c
                result[j, i] = result[i, j]
                common[i, j] = n_common
                common[j, i] = n_common
    return result, common
