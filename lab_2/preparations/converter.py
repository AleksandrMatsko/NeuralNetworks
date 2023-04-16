import pandas as pd
import numpy as np
import typing


def convert_characters_to_nums(df: pd.DataFrame, columns: typing.List[str],
                               nan_sym: str = "?") -> typing.Tuple[pd.DataFrame, dict, dict]:
    new_df = df.copy(deep=True)
    characters_to_nums = dict()
    nums_to_characters = dict()
    for col_name in columns:
        unq_vals = list(df[col_name].unique())
        if nan_sym in unq_vals:
            unq_vals.pop(unq_vals.index(nan_sym))
            unq_vals.append(nan_sym)
        col_characters_to_nums = dict()
        col_nums_to_characters = dict()
        for i in range(len(unq_vals)):
            if unq_vals[i] == nan_sym:
                col_characters_to_nums[nan_sym] = np.nan
                col_nums_to_characters[np.nan] = nan_sym
            else:
                col_characters_to_nums[unq_vals[i]] = i
                col_nums_to_characters[i] = unq_vals[i]
        new_df[col_name] = new_df[col_name].apply(lambda x: col_characters_to_nums.get(x), convert_dtype=False)
        characters_to_nums[col_name] = col_characters_to_nums
        nums_to_characters[col_name] = col_nums_to_characters
    return new_df, characters_to_nums, nums_to_characters

def convert_vals_with_rule(df: pd.DataFrame, columns: typing.List[str], replace_dict: dict) -> pd.DataFrame:
    new_df = df.copy(deep=True)
    for col_name in columns:
        value_dict = replace_dict.get(col_name)
        if value_dict is None:
            raise ValueError(f"No rule for column {col_name}")
        if not isinstance(value_dict, dict):
            raise ValueError(f"No rule for column {col_name}, got {type(value_dict)} excepted dict")
        unq_vals = list(df[col_name].unique())
        for unq_val in unq_vals:
            if value_dict.get(unq_val) is None and not np.isnan(unq_val):
                raise ValueError(f"No rule for value {unq_val} in column {col_name}")
        new_df[col_name] = new_df[col_name].apply(lambda x: value_dict.get(x))
    return new_df
