import re
import os
from typing import Union, Sequence, Hashable, Iterable

import pandas as pd

from . import cached_reader

_RANGE_RE = re.compile(r"\[(-?\d+),(-?\d+)[)\]]")  # Week regex

_CLALIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "pregnancy.1w")
_CLALIT_BY_AGE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "pregnancy.by_age.1w")
_METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "Metadata.csv")


def get_metadata(metadata_path: str = _METADATA_PATH) -> pd.DataFrame:
    """
    Mapping lab test names and units
    :param metadata_path: path to metadata csv
    :return: A DataFrame with the index column as full test name, and columns: "Short name", "LabNorm name", "Nice name", "Group", "units"
    """
    return cached_reader(metadata_path, index_col=0)


def get_condition_from_dirname(dir_name: str) -> str:
    """
    Try and find the condition (pathology) from the filename. Currently, supports pre-eclampsia, gdm, and postpartum hemorrhage.
    If found the names are mapped to the short names: Pre-ecl, GDM, PPH. If not found, returns an empty string.
    :param dir_name: The Clalit directory name.
    :return: Condition short name if found or empty string
    """
    m = re.match(r"^pregnancy\.(.\w*)\.?\dw$", dir_name)
    cond_dict = {"pre-eclampsia": "Pre-ecl",
                 "gdm": "GDM",
                 "postpartum_hemorrhage": "PPH"}
    if m is not None:
        return cond_dict.get(m.group(1), "")


def translate_long_to_short_lab(long_names: Union[Hashable, Sequence[Hashable]]) -> pd.Series:
    """
    Translate long lab names to short names. All the long names MUST be in the metadata index(valid lab test names).
    :param long_names:
    :return:
    """
    metadata = get_metadata()
    return metadata.loc[long_names, "Short name"]


def group_tests(tests: Iterable[str]) -> dict[str, list[str]]:
    """
    Group tests by the "Group" column in the metadata.
    :param tests: Iterable of test names which SHOULD be in the metadata index (valid lab test names).
    :return: Mapping from group name to valid lab test name
    """
    metadata = get_metadata()
    metadata = metadata.loc[metadata.index.intersection(tests)]
    return metadata.groupby("Group").groups


def parse_week_from_str(week_str: str) -> float:
    """
    Parse the week string to a float. The week string is in the format: "[start, end)".
    :param week_str: The week string.
    :return: The midpoint of the range.
    :raises: ValueError if the week string is of wrong format.
    """
    match = _RANGE_RE.match(week_str)
    if match is None:
        raise ValueError(f"Invalid week string: {week_str}")
    return (int(match.group(1)) + int(match.group(2))) / 2


def get_clalit_data(test_name: str, clalit_path: str = _CLALIT_PATH) -> pd.DataFrame:
    """
    Get the Clalit data for a specific test. The data is indexed by the week of pregnancy (midpoint of the range).
    :param test_name: Valid test name.
    :param clalit_path: Path to the Clalit data directory.
    :return: A DataFrame with the index as the week of pregnancy and columns, important among which are: "val_(q)mean", "val_(q)sd", "val_n", "val_min", "(q)val_(5,10,25,50,75,90,95)"
    """
    df = cached_reader(os.path.join(clalit_path, test_name + ".csv")).copy()
    df["week"] = df["week"].apply(parse_week_from_str)
    return df.set_index("week")

def get_data_by_tests_and_field(tests: Sequence[str], field: str):
    sers = []
    for test in tests:
        ser = get_clalit_data(test)[field]
        ser.name = test
        sers.append(ser)
    return pd.concat(sers, axis=1)


def get_quantiles_from_column_names(column_names: Iterable[str]) -> (list[float], list[str]):
    """
    Get the percentiles and convert to quantiles from the column names of each Clalit lab test CSV.
    :param column_names: The column names to parse.
    :return: The quantiles in the given order.
    :raises ValueError: If the column names are not in the expected format.
    """
    quantiles = []
    columns = []
    for column_name in column_names:
        match = re.match("^val_(\d+)$", column_name)  # Assumes the columns to use are val_mean and val_sd
        if match is not None:
            quant = int(match.group(1)) / 100
            quantiles.append(quant)
            columns.append(column_name)
    return quantiles, columns
