import os
import re
from typing import Union, Collection, Iterable, Generator, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import cached_reader

_LABNORM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "lab.quant.clalit")


def interp_per_age(lab: str, is_quantile: bool = True, labnorm_path: str = _LABNORM_PATH, female_only: bool = True) -> \
        dict[Union[tuple[int], tuple[int, str]], Callable[[np.ndarray], np.ndarray]]:
    """
    Map each group to a linear interpolator of its LabNorm reference values
    :param lab: Lab test name.
    :param is_quantile: If True, Interpolates quantiles from test values. If False, interpolates values from quantiles.
    :param labnorm_path: Path to LabNorm directory.
    :param female_only: If True, only use female reference values and the keys are just age. If false, keys are (age, gender) groups.
    :return: Mapping from group to the interpolator. The interpolator accepts either single numerical values or numpy arrays.
    """
    df = _query_labnorm(lab, labnorm_path, female_only)
    group_to_interpolator = {}
    for group, quant, val, _ in _iter_df_age_quant_vals(df, not female_only):
        x, y = (quant, val) if is_quantile else (val, quant)
        group_to_interpolator[group] = interp1d(x, y, bounds_error=False,
                                                fill_value=(0, y.max()))
    return group_to_interpolator


def age_to_quant_and_vals(lab: str, labnorm_path: str = _LABNORM_PATH, min_quant_step: float = 0.02,
                          ages: Union[Collection[int], None] = None, ignore_tail: int = 50) -> Union[
    dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Map age to its downsampled quantile and value DataFrame for the lab test.
    The downsampling is done by taking quantiles at regular intervals since some tests have reference too dense with quantiles.
    :param lab: Lab test name.
    :param labnorm_path: Path to a LabNorm directory. Uses default if None.
    :param min_quant_step: The minimal step between two consecutive quantiles in the  map
    :param ages: Specific ages to consider. If None, consider all ages [20-89]. If one age is given, returns the DataFrame and not a mapping
    :param ignore_tail: Number of individual measurements in the Clalit database to ignore at the tails. This avoids common errors in data records.
    :return: Mapping from age to DataFrame with columns "quant" and "value" for this lab test
    """
    df = _query_labnorm(lab, labnorm_path)
    downsampled_signal = _downsample_dist_per_age(df, min_quant_step, ages_to_consider=ages, ignore_tail=ignore_tail)
    if ages is not None and len(ages) == 1:
        return downsampled_signal.get(next(iter(ages)))
    return downsampled_signal


def find_median_for_lab(lab: str) -> pd.Series:
    """
    Mapping from age to median value for the lab test.
    :param lab: Lab test name
    :return: Index is the age, the values are median values of the lab test, name is the lab test.
    """
    age_to_median = {}
    for age, interp in interp_per_age(lab).items():
        try:
            age_to_median[age[0]] = interp(np.array([0.5])).item()
        except ValueError as e:
            print(f"Error in {lab} at age {age}: {e}")
            raise e
    return pd.Series(age_to_median, age_to_median.keys(), name=lab)


def get_unique_reference_ages_from_labnorm(lab_names: Iterable[str], labnorm_path: str = _LABNORM_PATH) -> np.ndarray:
    """
    Intersection of all reference ages in the lab names.
    :param lab_names: All the lab names to consider for the intersection.
    :param labnorm_path: Path to a LabNorm directory. Uses default if None.
    :return: An array of all ages in int format.
    """
    ages = None
    for lab in lab_names:
        df = _query_labnorm(lab, labnorm_path)
        lab_ages = np.unique(list(map(_parse_age_labnorm, df["age"].values)))
        if ages is None:
            ages = lab_ages
        else:
            ages = np.intersect1d(ages, lab_ages)
    return ages


def _parse_age_labnorm(age: str) -> int:
    # Return the bottom of the range. Throws an exception if match fails
    m = re.match(r"\[(\d{2,}),\s*\d{2,}\)", age)
    if m is None:
        raise ValueError(f"Invalid age format: {age}")
    return int(m[1])


def _query_labnorm(lab: str, labnorm_path: str, female_only=True):
    if female_only:
        return cached_reader(os.path.join(labnorm_path, lab + ".csv"), "gender=='female'")
    else:
        return cached_reader(os.path.join(labnorm_path, lab + ".csv"))


def _ignore_tail_sym(n, tail_ignore):
    # How many tests to ignore from the tails of the reference values
    if isinstance(tail_ignore, int) and n is not None:
        return tail_ignore / (2 * n)
    elif isinstance(tail_ignore, float):
        return tail_ignore / 2
    else:
        return 0.0


def _downsample_dist_per_age(labnorm_df: pd.DataFrame, quant_step: float = 0.02, upper_quant_step: float = 0.025,
                             ages_to_consider=Union[Collection[int], None],
                             ignore_tail: int = 50) -> dict[int, pd.DataFrame]:
    """
    Included indices only where the difference is greater than a certain quantile difference.
    :param labnorm_df: The raw LabNorm DataFrame from age to quantile and value. Gender MUST be unique.
    :param quant_step: Minimal step, less than that is ignored, unless the next value has a difference too great
    :param upper_quant_step: Never ignore a value if the difference is more than this. MUST be greater than `quant_step`
    :param ages_to_consider: Consider only these ages, or all if None
    :param ignore_tail: Ignore these many individuals in the original dataset in the tails as they are likely to be errors.
    :return: Mapping from age to DataFrame with columns "quant" and "value" for this lab test after downsampling.
    """
    age_to_quant_val_df = {}
    for g, quant, val, n in _iter_df_age_quant_vals(labnorm_df):
        age = g[0]
        if ages_to_consider is not None and age not in ages_to_consider:
            continue
        quant = quant.to_numpy()
        val = val.to_numpy()
        ignore_tail_sym = _ignore_tail_sym(n, ignore_tail)
        first_ind = (quant > ignore_tail_sym).argmax()  # Find first True value
        last_ind = np.nonzero(quant < (1 - ignore_tail_sym))[0][-1]  # Find last True value
        quant = quant[first_ind:last_ind + 1]
        val = val[first_ind:last_ind + 1]
        indices = [0]  # Use the first index, and start from here.
        last_value = quant[0]  # Initialize the last quantile considered
        for i in range(1, len(quant)):
            if quant[i] - last_value < quant_step:  # Don't include less than this much difference
                continue
            elif quant[i] - last_value <= upper_quant_step:  # Include if difference is not too great
                indices.append(i)
                last_value = quant[i]
            # If difference is too great, include previous index (as long as it does not equal the last value, implying it is already included).
            elif (quant[i] - last_value > upper_quant_step) and (quant[i - 1] > last_value):
                indices.append(i - 1)
                last_value = quant[i - 1]
            else:
                indices.append(i)
                last_value = quant[i]
        indices = np.array(indices)
        age_to_quant_val_df[age] = pd.DataFrame(
            {"quant": quant[indices], "value": val[indices]})
    return age_to_quant_val_df


def _iter_df_age_quant_vals(df: pd.DataFrame, groupby_gender: bool = False) -> Generator[tuple[
    Union[tuple[int], tuple[int, str]], pd.Series, pd.Series, Union[int, None]], None, None]:
    """
    Extract the quantiles and respecting values for a LabNorm reference per group. Group is either age or age and gender.
    If grouping also by gender (recommended if the dataframe has male and female), the
    :param df: DataFrame with columns "age", "quant", "value", "n". If `groupy_gender` is True, MUST also have `gender` column
    :param groupby_gender: If True, group by gender. If False, group by age only.
    :return: A generator with the group as the first element, the quantiles as the second, the values as the third and the `n` as the fourth. The `n` value is unqiue per group.
    """
    if groupby_gender:
        groupby = ["age", "gender"]
    else:
        groupby = ["age"]
    for group, group_df in df.groupby(groupby):
        group: list[str] = list(group)  # tuple is immutable
        age = _parse_age_labnorm(group[0])  # age is the first element
        group: Union[tuple[int], tuple[int, str]] = tuple([age] + group[1:])
        n: Union[int, None] = int(group_df["n"].iloc[0]) if "n" in group_df.columns else None  # n value is unique per group, take the first arbitrarily. If not present, set to None
        val: pd.Series = group_df["value"]
        quant: pd.Series = group_df["quant"]
        yield group, quant, val, n
