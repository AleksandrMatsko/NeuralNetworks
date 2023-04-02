import typing

import numpy as np
import pandas as pd


def split_dataset(df: pd.DataFrame, learn_percent: int, target_columns: list) -> typing.Tuple:
    if learn_percent < 0 or learn_percent > 100:# or learn_percent < 50:
        return None, None
    no_nans = df[(~np.isnan(df[target_columns]).any(axis=1)) & ~np.isnan(df[target_columns]).all(axis=1)]
    some_nans = df[(~np.isnan(df[target_columns])).any(axis=1) & (np.isnan(df[target_columns])).any(axis=1)]
    num_in_learn_no_nans = int(len(no_nans.index) / 100 * learn_percent)
    num_in_learn_some_nans = int(len(some_nans.index) / 100 * learn_percent)
    learn_no_nans = no_nans.sample(num_in_learn_no_nans)
    learn_some_nans = some_nans.sample(num_in_learn_some_nans)
    test_no_nans = no_nans.merge(learn_no_nans, how='left', indicator=True) \
        .query("_merge == 'left_only'").drop('_merge', axis=1)[no_nans.columns]
    test_some_nans = some_nans.merge(learn_some_nans, how='left', indicator=True) \
        .query("_merge == 'left_only'").drop('_merge', axis=1)[some_nans.columns]
    learn = pd.concat([learn_no_nans, learn_some_nans], ignore_index=True)
    test = pd.concat([test_no_nans, test_some_nans], ignore_index=True)
    return learn.sample(n=len(learn.index)).reset_index(drop=True), test
