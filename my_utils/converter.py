import pandas as pd
import numpy as np
import typing


def convert_characters_to_nums(df: pd.DataFrame, nan_sym: str) -> typing.Tuple[pd.DataFrame, dict]:
    new_df = df.copy(deep=True)
    feature_replacement = dict()
    for col_name in df.columns:
        unq_vals = list(df[col_name].unique())
        if nan_sym in unq_vals:
            unq_vals.pop(unq_vals.index(nan_sym))
            unq_vals.append(nan_sym)
        col_replace = dict()
        for i in range(len(unq_vals)):
            if unq_vals[i] == nan_sym:
                col_replace[nan_sym] = np.nan
            else:
                col_replace[unq_vals[i]] = i
        new_df[col_name] = new_df[col_name].apply(lambda x: col_replace.get(x), convert_dtype=False)
        feature_replacement[col_name] = col_replace
    return new_df, feature_replacement