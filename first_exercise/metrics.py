__author__ = 'massimo'
import numpy as np
import unittest

def roc_auc(ranked_list, pos_items):
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    ranks = np.arange(len(ranked_list))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0
    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])
    assert 0 <= auc_score <= 1
    return auc_score


def precision(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(ranked_list)
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]
    assert 0 <= recall_score <= 1
    return recall_score

def rr(ranked_list, pos_items, at=None):
    # reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    ranks = np.arange(1, len(ranked_list)+1)[is_relevant]
    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0

def map(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([pos_items.shape[0], len(ranked_list)])
    assert 0 <= map_score <= 1
    return map_score


def ndcg(ranked_list, pos_items, relevance=None, at=None):
    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)
    ideal_dcg = dcg(np.sort(relevance)[::-1])
    rank_dcg = dcg(rank_scores)
    ndcg = rank_dcg / ideal_dcg
    assert 0 <= ndcg <= 1
    return ndcg


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)))


metrics = ['AUC', 'Precision' 'Recall', 'MAP', 'NDCG']

def pp_metrics(metric_names, metric_values, metric_at):
    """
    Pretty-prints metric values
    :param metrics_arr:
    :return:
    """
    assert len(metric_names) == len(metric_values)
    if isinstance(metric_at, int):
        metric_at = [metric_at] * len(metric_values)
    return ' '.join(['{}: {:.4f}'.format(mname, mvalue) if mcutoff is None or mcutoff == 0 else
                     '{}@{}: {:.4f}'.format(mname, mcutoff, mvalue)
                     for mname, mcutoff, mvalue in zip(metric_names, metric_at, metric_values)])


class TestAUC(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4])
        ranked_list = np.asarray([1, 2, 3, 4, 5])
        self.assertTrue(np.allclose(roc_auc(ranked_list, pos_items),
                                    (2. / 3 + 1. / 3) / 2))


class TestRecall(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(recall(ranked_list_1, pos_items), 3. / 4))
        self.assertTrue(np.allclose(recall(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(recall(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1./4, 1./4, 2./4, 3./4]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(recall(ranked_list_1, pos_items, at=at)), val))

class TestPrecision(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(precision(ranked_list_1, pos_items), 3. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_2, pos_items), 4. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1./2, 1./3, 2./4, 3./5]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(precision(ranked_list_1, pos_items, at=at)), val))


class TestRR(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(rr(ranked_list_1, pos_items), 1. / 2))
        self.assertTrue(np.allclose(rr(ranked_list_2, pos_items), 1.))
        self.assertTrue(np.allclose(rr(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1./2, 1./2, 1./2, 1./2]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(rr(ranked_list_1, pos_items, at=at)), val))



class TestMAP(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        ranked_list_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        ranked_list_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])
        self.assertTrue(np.allclose(map(ranked_list_1, pos_items), (1. / 2 + 2. / 4 + 3. / 5) / 4))
        self.assertTrue(np.allclose(map(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(map(ranked_list_3, pos_items), 0.0))
        self.assertTrue(np.allclose(map(ranked_list_4, pos_items), (1./7 + 2./8 + 3./9 + 4./10) / 4))
        self.assertTrue(np.allclose(map(ranked_list_5, pos_items), (1. + 2./7 + 3./8 + 4./9) / 4))

        thresholds = [1, 2, 3, 4, 5]
        values = [
            0.0,
            1. / 2 / 2,
            1. / 2 / 3,
            (1. / 2 + 2. / 4) / 4,
            (1. / 2 + 2. / 4 + 3. / 5) / 4
            ]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(map(ranked_list_1, pos_items, at)), val))



class TestNDCG(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        pos_relevances = np.asarray([5, 4, 3, 2])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
        idcg = ((2 ** 5 - 1) / np.log(2) +
                (2 ** 4 - 1) / np.log(3) +
                (2 ** 3 - 1) / np.log(4) +
                (2 ** 2 - 1) / np.log(5))
        self.assertTrue(np.allclose(dcg(np.sort(pos_relevances)[::-1]), idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_1, pos_items, pos_relevances),
                                    ((2 ** 5 - 1) / np.log(3) +
                                     (2 ** 4 - 1) / np.log(5) +
                                     (2 ** 3 - 1) / np.log(6)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_2, pos_items, pos_relevances),
                                    ((2 ** 2 - 1) / np.log(2) +
                                     (2 ** 3 - 1) / np.log(3) +
                                     (2 ** 5 - 1) / np.log(4) +
                                     (2 ** 4 - 1) / np.log(5)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_3, pos_items, pos_relevances), 0.0))


if __name__ == '__main__':
    unittest.main()
