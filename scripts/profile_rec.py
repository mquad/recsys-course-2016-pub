import pstats, cProfile
import scipy.sparse as sps
import numpy as np
from recpy.recommenders.mf import AsySVD as Recommender
# from recpy.recommenders.mf import FunkSVD as Recommender

np.random.seed(1)
X = sps.rand(1000, 2000, format='csr', density=0.01)
print(X.nnz)
rec = Recommender(iters=1)
cProfile.runctx("rec.fit(X)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
