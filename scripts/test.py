import numpy as np
import scipy.sparse as sps
from datetime import datetime as dt

def cols_norm_1(csc_matrix):
    csc_sq = csc_matrix.copy()
    csc_sq.data **= 2
    norm = np.sqrt(csc_sq.sum(axis=0))
    return np.asarray(norm).ravel()

def cols_norm_2(csc_matrix):
    vec = np.cumsum(csc_matrix.data ** 2)
    col_sums = vec[csc_matrix.indptr[1:]-1]
    norm = np.r_[col_sums[0], np.diff(col_sums)]
    return np.sqrt(norm)

# 1000 x 2000, 1% sparsity
X_small = sps.random(1000, 2000, 0.01, format='csc')
# 100000 x 10000, 1% sparsity
X_large = sps.random(100000, 10000, 0.01, format='csc')

# check for correctness
print(np.allclose(cols_norm_1(X_small), cols_norm_2(X_small)))

# benckmarks
t0 = dt.now()
cols_norm_1(X_small)
print('cols_norm_1(X_small): {}'.format(dt.now() - t0)) # 0.4ms

t0 = dt.now()
cols_norm_2(X_small)
print('cols_norm_2(X_small): {}'.format(dt.now() - t0)) # 0.3ms

t0 = dt.now()
cols_norm_1(X_large)
print('cols_norm_1(X_large): {}'.format(dt.now() - t0)) # 0.11s

t0 = dt.now()
cols_norm_2(X_large)
print('cols_norm_2(X_large): {}'.format(dt.now() - t0)) # 0.26s
