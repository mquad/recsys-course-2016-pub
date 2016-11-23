import numpy as np


def holdout(data, user_key='user_id', item_key='item_id', perc=0.8, seed=1234, clean_test=True):
    # set the random seed
    rng = np.random.RandomState(seed)
    #  shuffle data
    nratings = data.shape[0]
    shuffle_idx = rng.permutation(nratings)
    train_size = int(nratings * perc)
    # split data according to the shuffled index and the holdout size
    train_split = data.ix[shuffle_idx[:train_size]]
    test_split = data.ix[shuffle_idx[train_size:]]

    # remove new user and items from the test split
    if clean_test:
        train_users = train_split[user_key].unique()
        train_items = train_split[item_key].unique()
        test_split = test_split[(test_split[user_key].isin(train_users)) & (test_split[item_key].isin(train_items))]

    return train_split, test_split


def k_fold_cv(data, user_key='user_id', item_key='item_id', k=5, seed=1234, clean_test=True):
    # set the random seed
    rng = np.random.RandomState(seed)
    # shuffle data
    nratings = data.shape[0]
    shuffle_idx = rng.permutation(nratings)
    fold_size = -(-nratings // k)

    for fidx in range(k):
        train_idx = np.concatenate([shuffle_idx[:fidx * fold_size], shuffle_idx[(fidx + 1) * fold_size:]])
        test_idx = shuffle_idx[fidx * fold_size:(fidx + 1) * fold_size]
        train_split = data.ix[train_idx]
        test_split = data.ix[test_idx]

        # remove new user and items from the test split
        if clean_test:
            train_users = train_split[user_key].unique()
            train_items = train_split[item_key].unique()
            test_split = test_split[(test_split[user_key].isin(train_users)) & (test_split[item_key].isin(train_items))]
        yield train_split, test_split
