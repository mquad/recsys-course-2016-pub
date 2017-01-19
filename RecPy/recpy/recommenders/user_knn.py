from .item_knn import ItemKNNRecommender
import numpy as np


class UserKNNRecommender(ItemKNNRecommender):
    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        super().__init__(
            k=k,
            shrinkage=shrinkage,
            similarity=similarity,
            normalize=normalize,
            sparse_weights=sparse_weights
        )

    def __str__(self):
        return "UserKNN(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        M, N = X.shape
        Xt = X.T.tocsr()
        # fit a ItemKNNRecommender on the transposed X matrix
        super().fit(Xt)
        self.dataset = X
        # precompute the predicted scores for speed
        if self.sparse_weights:
            self.scores = self.W_sparse.dot(X).toarray()
        else:
            self.scores = self.W.dot(X)
        if self.normalize:
            for i in range(M):
                rated = Xt[i].copy()
                rated.data = np.ones_like(rated.data)
                if self.sparse_weights:
                    den = rated.dot(self.W_sparse).toarray().ravel()
                else:
                    den = rated.dot(self.W).ravel()
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                self.scores[:, i] /= den

    def recommend(self, user_id, n=None, exclude_seen=True):
        ranking = self.scores[user_id].argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]
