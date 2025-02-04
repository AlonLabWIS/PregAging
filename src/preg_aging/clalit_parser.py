import re
import os
from typing import Union, Sequence, Hashable

import numpy as np
import pandas as pd

from . import cached_reader

_RANGE_RE = re.compile(r"\[(-?\d+),(-?\d+)[)\]]")

_CLALIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "pregnancy.1w")
_CLALIT_BY_AGE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "pregnancy.by_age.1w")
_METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "Metadata.csv")


def get_metadata(metadata_path=_METADATA_PATH):
    return cached_reader(metadata_path, index_col=0)

def get_condition_from_filename(file_name):
    m = re.match(r"^pregnancy\.(.\w*)\.?\dw$", file_name)
    cond_dict = {"pre-eclampsia": "Pre-ecl",
                 "gdm": "GDM",
                 "postpartum_hemorrhage": "PPH"}
    if m is not None:
        return cond_dict.get(m.group(1), "")

def translate_long_to_short_lab(long_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    metadata = get_metadata()
    return metadata.loc[long_names, "Short name"]

def translate_long_to_labnorm_name(short_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    metadata = get_metadata()
    return metadata.loc[short_names, "LabNorm name"]

def translate_long_to_nice_name(short_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    metadata = get_metadata()
    return metadata.loc[short_names, "Nice name"]

def group_tests(tests:list[str] = None):
    metadata = get_metadata()
    if tests is not None:
        metadata = metadata.loc[metadata.index.intersection(tests)]
    return metadata.groupby("Group").groups


def get_clalit_test_names():
    return list(map(lambda x: os.path.splitext(os.path.basename(x))[0], os.listdir(_CLALIT_PATH)))


def get_clalit_data(test_name, clalit_path=_CLALIT_PATH):
    df = cached_reader(os.path.join(clalit_path, test_name + ".csv")).copy()
    df["week"] = df["week"].apply(parse_week_from_str)
    return df.set_index("week")


def parse_week_from_str(week_str: str) -> float:
    match = _RANGE_RE.match(week_str)
    if match is None:
        raise ValueError(f"Invalid week string: {week_str}")
    return (int(match.group(1)) + int(match.group(2))) / 2


def get_clalit_age_data(test_name):
    df = cached_reader(os.path.join(_CLALIT_BY_AGE_PATH, test_name + ".csv")).copy().iloc[:, :30]
    df["week"] = df["week"].apply(parse_week_from_str)
    df["age_group"] = df["age_group"].apply(parse_week_from_str)
    return df.set_index(["week", "age_group"])


def filter_clalit_data_by_weeks(df: pd.DataFrame, min_week: int = None, max_week: int = None):
    return df.loc[min_week:max_week]


def mean_by_col(df, int_column_for_mean: str = "val_n"):
    mean_col = df[int_column_for_mean]
    df_no_mean_col = df.drop(columns=[int_column_for_mean])
    weighted_df = (df_no_mean_col.mul(mean_col, axis=0).sum() / mean_col.sum())
    weighted_df[int_column_for_mean] = int(mean_col.sum())
    return weighted_df.to_frame().T


def get_mean_age_pre_conception(test_name, min_week=-60, max_week=-40):
    df = get_clalit_age_data(test_name)
    df = filter_clalit_data_by_weeks(df, min_week, max_week)
    grouped = df.groupby(level="age_group")
    return grouped.apply(mean_by_col).droplevel(None)

def get_data_by_tests_and_field(tests: Sequence[str], field: str):
    sers = []
    for test in tests:
        ser = get_clalit_data(test)[field]
        ser.name = test
        sers.append(ser)
    return pd.concat(sers, axis=1)


def recalculate_mean_sd_lin_approx(quantiles, values):
    """
    Piecewise linear approximation to mean and standard deviation.
    :param quantiles: Length n >1
    :param values: Same length as quantiles
    :return:
    """
    # Find the left and right support for the CDF
    left_extrapolation_x = (values[:, 0], values[:, 1])  # any point left of the min would do
    left_extrapolation_y = (quantiles[0], quantiles[1])
    right_extrapolation_x = (values[:, -2], values[:, -1])  # any point right of the max would do
    right_extrapolation_y = (quantiles[-2], quantiles[-1])
    left_support = _find_target_linear(*left_extrapolation_x, *left_extrapolation_y, target=0)
    right_support = _find_target_linear(*right_extrapolation_x, *right_extrapolation_y, target=1)
    segment_edges = np.hstack((left_support[:, np.newaxis], values, right_support[:, np.newaxis]))
    right_edges = segment_edges[:, 1:]
    left_edges = segment_edges[:, :-1]
    slopes = _find_slopes_pieces(values, quantiles)
    # Mean is definite integral of piecewise linear function, slopes*x
    mean = ((right_edges ** 2 - left_edges ** 2) * slopes / 2).sum(axis=1)
    # SD is sqrt of definite integral of piecewise quadratic function, slopes*x**2
    sd = np.sqrt(((right_edges ** 3 - left_edges ** 3) * slopes / 3).sum(axis=1) - mean ** 2)
    return mean, sd


def recalculate_mean_sd_bad_tests(test_name, test_df: pd.DataFrame,
                                  test_sem_thresholds={"BMI": 0.3, "CK_CREAT": 10,
                                                       "LACTIC_DEHYDROGENASE_LDH__BLOOD": 4.5},
                                  sd_column="val_sd", n_column="val_n", mean_column="val_mean"):
    test_df = test_df.copy()
    if test_name not in test_sem_thresholds:
        return test_df
    sem = test_df[sd_column] / test_df[n_column] ** 0.5
    # TODO argument where sem is bad
    bad_sem_indices = sem.index[sem > test_sem_thresholds[test_name]]
    quantiles, columns = _get_quantiles_from_column_names(test_df.columns)
    sorted_quantiles = np.argsort(quantiles)
    columns = np.array(columns)[sorted_quantiles]
    quantiles = np.array(quantiles)[sorted_quantiles]
    mean, sd = recalculate_mean_sd_lin_approx(quantiles, test_df.loc[bad_sem_indices, columns].values)
    test_df.loc[bad_sem_indices, [mean_column, sd_column]] = np.stack([mean, sd], axis=1)
    return test_df

def join_weekly_bins(test_name, num_bins_to_join=2):
    df = get_clalit_age_data(test_name)
    if num_bins_to_join == 1:
        return df



def _find_target_linear(x1, x2, y1, y2, target):
    slope = (y2 - y1) / (x2 - x1)
    return (target - y1) / slope + x1


def _find_slopes_pieces(xs: np.ndarray, ys: np.ndarray):
    slopes = (ys[1:] - ys[:-1]) / (xs[:, 1:] - xs[:, :-1])
    return np.pad(slopes, ((0, 0),(1, 1)), mode="edge")


def _get_quantiles_from_column_names(column_names):
    quantiles = []
    columns = []
    for column_name in column_names:
        match = re.match("^val_(\d+)$", column_name)  # Assumes the columns to use are val_mean and val_sd
        if match is not None:
            quant = int(match.group(1)) / 100
            quantiles.append(quant)
            columns.append(column_name)
    return quantiles, columns
