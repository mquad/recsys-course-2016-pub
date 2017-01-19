import numpy as np
import pandas as pd


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


def split_by_user(data, user_key='user_id', item_key='item_id', split_ids=None, perc=0.8, seed=1234,
                  compress_user_indices=True):
    # set the random seed
    rng = np.random.RandomState(seed)
    if split_ids is not None:
        assert len(split_ids) == 2
        train_users, test_users = split_ids
    else:
        # partition users into two groups
        users = data[user_key].unique()
        users_shuffled = rng.permutation(users)
        train_size = int(len(users) * perc)
        train_users = users_shuffled[:train_size]
        test_users = users_shuffled[train_size:]

    train_split = data.ix[data[user_key].isin(train_users)]
    test_split = data.ix[data[user_key].isin(test_users)]
    if compress_user_indices:
        # TODO: fix the SettingWithCopyWarning due to reassignment of 'user_idx'
        # compress the user indices in train and test
        train_user_to_idx = pd.Series(index=train_users, data=np.arange(len(train_users)))
        train_split.loc[:, 'user_idx'] = train_user_to_idx[train_split[user_key]].values

        test_user_to_idx = pd.Series(index=test_users, data=np.arange(len(test_users)))
        test_split.loc[:, 'user_idx'] = test_user_to_idx[test_split[user_key]].values
        return train_split, test_split, train_user_to_idx, test_user_to_idx
    else:
        return train_split, test_split


def per_user_holdout(data, user_key='user_id', item_key='item_id', n_observed=1, seed=1234):

    # set the random seed
    rng = np.random.RandomState(seed)
    # sample `n_observed` ratings from each user at random
    observed, hidden = [], []
    grouped = data.groupby(user_key)
    for user, g in grouped:
        gsize = g.shape[0]
        idx_shuffled = rng.permutation(gsize)
        # sample observed and hidden ratings
        for i, idx in enumerate(idx_shuffled):
            if i < n_observed:
                observed.append(g.values[idx])
            else:
                hidden.append(g.values[idx])
    # build the DataFrames
    columns = data.columns
    observed_split = pd.DataFrame.from_records(observed, columns=columns)
    hidden_split = pd.DataFrame.from_records(hidden, columns=columns)
    return observed_split, hidden_split
