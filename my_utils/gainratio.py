import pandas as pd
import math


def info(df: pd.DataFrame, sub_df: pd.DataFrame, target_name: str) -> float:
    if df[target_name].nunique() == 0:
        return 0
    interval_count = 1 + int(math.log(df[target_name].nunique(), 2))
    left_border = df[target_name].min()
    right_border = df[target_name].max()
    step = (right_border - left_border) / interval_count

    vals_in_interval = [0] * interval_count
    value_counts = sub_df[target_name].value_counts()
    unq_vals = list(value_counts.index)
    if step != 0:
        for i in range(len(unq_vals)):
            interval_index = int((unq_vals[i] - left_border) // step)
            if interval_index == interval_count:
                interval_index -= 1
            vals_in_interval[interval_index] += value_counts[unq_vals[i]]

    s = 0.0
    for val in value_counts:
        tmp = val / sub_df[target_name].count()
        if tmp != 0:
            s += tmp * math.log(tmp, 2)
    return -s


def info_a(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    if df[attr_name].nunique() == 0:
        return 0
    interval_count = 1 + int(math.log(df[attr_name].nunique(), 2))
    left_border = df[attr_name].min()
    right_border = df[attr_name].max()
    step = (right_border - left_border) / interval_count
    left_borders = []
    for i in range(interval_count):
        left_borders.append(left_border + i * step)

    s = 0.0
    for border in left_borders:
        sub_df = df.loc[((df[attr_name] >= border) & (df[attr_name] < border + step)) |
                        (df[attr_name] == right_border)]
        s += info(df, sub_df, target_name)
    return s


def split_info(df: pd.DataFrame, attr_name: str) -> float:
    s = 0.0
    for val in df[attr_name].value_counts():
        tmp = val / df[attr_name].count()
        s += tmp * math.log(tmp, 2)
    return s


def gain(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    return info(df, df, target_name) - info_a(df, attr_name, target_name)


def gain_ratio(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    return gain(df, attr_name, target_name) / split_info(df, attr_name)


def data_set_gain_ratio(df: pd.DataFrame, target_name: str, num_target_columns: int) -> pd.Series:
    gain_ratio_list = []
    for col_name in df.columns[0:-num_target_columns]:
        gain_ratio_list.append(gain_ratio(df, col_name, target_name))
    return pd.Series(gain_ratio_list, index=df.columns[0:-num_target_columns])
