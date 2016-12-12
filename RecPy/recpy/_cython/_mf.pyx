# cython: profile=True
# cython: linetrace=True
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

@cython.boundscheck(False)
def FunkSVD_sgd(R, num_factors=50, lrate=0.01, reg=0.015, iters=10, init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    # in csr format, indices correspond to column indices
    # let's build the vector of row_indices
    cdef np.ndarray[np.int64_t, ndim=1] row_nnz = np.diff(indptr).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] row_indices = np.repeat(np.arange(M), row_nnz).astype(np.int64)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)

    #
    # TODO: learn the U and V factors of FunkSVD with SGD
    #

    return U, V

@cython.boundscheck(False)
def AsySVD_sgd(R, num_factors=50, lrate=0.01, reg=0.015, iters=10, init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    #
    # TODO: learn the X and Y factors of AsySVD with SGD
    #
    return X, Y

@cython.boundscheck(False)
def AsySVD_compute_user_factors(user_profile, Y):
    if not isinstance(user_profile, sps.csr_matrix):
        raise ValueError('user_profile must be an instance of scipy.sparse.csr_matrix')
    assert user_profile.shape[0] == 1, 'user_profile must be a 1-dimensional vector'

    # use Cython MemoryViews for fast access to the sparse structure of user_profile
    cdef int [:] col_indices = user_profile.indices
    cdef float [:] data = user_profile.data

    # intialize the accumulated user profile
    cdef int num_factors = Y.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] Y_acc = np.zeros(num_factors, dtype=np.float32)
    cdef int n_rated = len(col_indices)
    # aux variables
    cdef int n
    # accumulate the item vectors for the items rated by the user
    for n in range(n_rated):
        ril = data[n]
        Y_acc += ril * Y[col_indices[n]]
    if n_rated > 0:
        Y_acc /= np.sqrt(n_rated)
    return Y_acc

