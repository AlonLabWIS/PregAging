import re
import os
from typing import Union, Sequence, Hashable

import pandas as pd

from . import cached_reader

# from .analyses import recalculate_mean_sd_lin_approx  # TODO: circular import with analyses.py

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


def translate_long_to_labnorm_name(long_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    metadata = get_metadata()
    return metadata.loc[long_names, "LabNorm name"]


def translate_long_to_nice_name(long_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    metadata = get_metadata()
    return metadata.loc[long_names, "Nice name"]


def group_tests(tests: list[str] = None):
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


def join_weekly_bins(test_name, num_bins_to_join=2):
    df = get_clalit_age_data(test_name)
    if num_bins_to_join == 1:
        return df


def get_quantiles_from_column_names(column_names):
    quantiles = []
    columns = []
    for column_name in column_names:
        match = re.match("^val_(\d+)$", column_name)  # Assumes the columns to use are val_mean and val_sd
        if match is not None:
            quant = int(match.group(1)) / 100
            quantiles.append(quant)
            columns.append(column_name)
    return quantiles, columns
