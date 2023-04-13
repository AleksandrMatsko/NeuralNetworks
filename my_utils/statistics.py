import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typing


def count_lines(df: pd.DataFrame) -> int:
    return len(df.index)


def count_non_nulls(df: pd.DataFrame) -> pd.Series:
    ser = df.count()
    ser.name = 'Количество'
    return ser


def count_percent_nulls(df: pd.DataFrame) -> pd.Series:
    num_lines = count_lines(df)
    ser = (num_lines - count_non_nulls(df)) / num_lines * 100
    ser.name = 'Процент_пропусков'
    return ser


def get_minimum(df: pd.DataFrame) -> pd.Series:
    ser = df.min(skipna=True)
    ser.name = 'Минимум'
    return ser


def get_maximum(df: pd.DataFrame) -> pd.Series:
    ser = df.max(skipna=True)
    ser.name = 'Максимум'
    return ser


def get_quantile(df: pd.DataFrame, q: float, quantile_name: str) -> pd.Series:
    Q = df.quantile(q=q)
    if quantile_name is not None and len(quantile_name) != 0:
        Q.name = quantile_name
    return Q


def get_standard_deviation(df: pd.DataFrame) -> pd.Series:
    ser = df.std()
    ser.name = 'Стандартное_отклонение'
    return ser


def get_average(df: pd.DataFrame) -> pd.Series:
    ser = df.mean(skipna=True)
    ser.name = 'Среднее'
    return ser


def get_median(df: pd.DataFrame) -> pd.Series:
    ser = df.median(skipna=True)
    ser.name = 'Медиана'
    return ser


def get_power(df: pd.DataFrame) -> pd.Series:
    ser = df.nunique()
    ser.name = 'Мощность'
    return ser


def get_interquartile_range(df: pd.DataFrame) -> pd.Series:
    ser = get_quantile(df, q=0.75, quantile_name='') - get_quantile(df, q=0.25, quantile_name='')
    ser.name = 'Интерквартильный_размах'
    return ser


def get_some_statistics(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.concat([
        count_non_nulls(df),
        count_percent_nulls(df),
        get_minimum(df),
        get_quantile(df, q=0.25, quantile_name='Первый_квартиль'),
        get_average(df),
        get_median(df),
        get_quantile(df, q=0.75, quantile_name='Третий_квартиль'),
        get_maximum(df),
        get_standard_deviation(df),
        get_power(df),
        get_interquartile_range(df)], axis=1)
    return f.T

def get_mode(df: pd.DataFrame) -> pd.Series:
    ser = df.mode(axis=0).iloc[0]
    ser.name = 'Мода'
    return ser

def get_percent_mode(df: pd.DataFrame, calc_mode_function: typing.Callable[[pd.DataFrame], pd.Series]) -> pd.Series:
    modes = calc_mode_function(df).T
    percents = dict()
    for column_name in df.columns:
        percents[column_name] = count_lines(df[df[column_name] == modes[column_name]]) / count_lines(df) * 100
    return pd.Series(percents).rename('Процент_моды')

def get_second_mode(df: pd.DataFrame) -> pd.Series:
    modes = get_mode(df).T
    no_modes_list = list()
    for column_name in df.columns:
        ser = df[df[column_name] != modes[column_name]][column_name]
        ser.name = column_name
        no_modes_list.append(ser)

    modes = list()
    for i in range(len(df.columns)):
        mode_dict = no_modes_list[i].mode().to_dict()
        try:
            modes.append(mode_dict[0])
        except KeyError:
            modes.append(np.nan)
    return pd.Series(data=modes, index=df.columns).rename('Вторая_мода')

def get_some_statistics_categorial(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.concat([
        count_non_nulls(df),
        count_percent_nulls(df),
        get_power(df),
        get_mode(df),
        get_percent_mode(df, get_mode),
        get_second_mode(df),
        get_percent_mode(df, get_second_mode)
    ], axis=1)
    return f.T

