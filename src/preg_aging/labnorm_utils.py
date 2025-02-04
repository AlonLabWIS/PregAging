import os
import re
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.ndimage import median_filter

from . import cached_reader

_LABNORM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "lab.quant.clalit")


def list_labnorm_tests(labnorm_path=_LABNORM_PATH):
    return [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(labnorm_path) if f.endswith("csv")]


def interp_per_age(lab, is_quantile=True, labnorm_path=_LABNORM_PATH, female_only=True):
    df = _query_labnorm(lab, labnorm_path, female_only)
    group_to_interpolator = {}
    for group, quant, val, _ in _iter_df_age_quant_vals(df, not female_only):
        x, y = (quant, val) if is_quantile else (val, quant)
        group_to_interpolator[tuple(group)] = interp1d(x, y, bounds_error=False,
                                                       fill_value=(0, y.max()))
    return group_to_interpolator


def age_to_quant_and_vals(lab, labnorm_path=_LABNORM_PATH, min_quant_step=0.02, ages=None, ignore_tail=50):
    df = _query_labnorm(lab, labnorm_path)
    downsampled_signal = _downsample_dist_per_age(df, min_quant_step, ages_to_consider=ages, ignore_tail=ignore_tail)
    if ages is not None and len(ages) == 1:
        return downsampled_signal.get(ages[0])
    return downsampled_signal


def find_median_for_lab(lab):
    age_to_median = {}
    for age, interp in interp_per_age(lab).items():
        try:
            age_to_median[age[0]] = interp(0.5).item()
        except ValueError as e:
            print(f"Error in {lab} at age {age}: {e}")
            raise e
    return pd.Series(age_to_median, age_to_median.keys(), name=lab)


def get_unique_reference_ages_from_labnorm(lab_names, labnorm_path=_LABNORM_PATH) -> np.ndarray:
    """
    Intersection of all reference ages in the lab names.
    :return:
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


def get_mean_std_per_test(test_name, female_only=True, smoothing_width=7):
    df = _query_labnorm(test_name, _LABNORM_PATH, female_only)
    mean = []
    std = []
    ind = []
    for i, (g, q, v, n) in enumerate(_iter_df_age_quant_vals(df, not female_only)):
        if n is None or n >= 200:
            if n is None:
                qs = q.values
                vals = median_filter(v.values, size=smoothing_width, mode="mirror")
                norm_factor = 1
            else:
                tail_exclusion = 50 / n  # min(50 / n, 0.005)
                norm_factor = (1 - 2 * tail_exclusion)
                q_min_arg = (q > tail_exclusion).idxmax()
                q_max_arg = (q < 1 - tail_exclusion).idxmin()
                qs = q.loc[q_min_arg:q_max_arg].values
                vals = v.loc[q_min_arg:q_max_arg].values
            uncorrected_mean = trapezoid(vals, qs)
            uncorrected_std = (trapezoid(vals ** 2, qs) - uncorrected_mean ** 2) ** 0.5
            std.append(uncorrected_std / norm_factor)
            mean.append(uncorrected_mean / norm_factor)
        else:  # n < 200
            mean.append(np.nan)
            std.append(np.nan)
        ind.append(g)
    if len(ind[0]) > 1:
        return pd.DataFrame({"mean": mean, "std": std}, index=pd.MultiIndex.from_tuples(ind))
    else:
        return pd.DataFrame({"mean": mean, "std": std}, index=tuple(map(lambda x: x[0], ind)))


def _parse_age_labnorm(age: str):
    return int(re.match(r"\[(\d{2,}),\s*\d{2,}\)", age)[1])


def _query_labnorm(lab, labnorm_path, female_only=True):
    if female_only:
        return cached_reader(os.path.join(labnorm_path, lab + ".csv"), "gender=='female'")
    else:
        return cached_reader(os.path.join(labnorm_path, lab + ".csv"))


def _ignore_tail_sym(n, tail_ignore):
    if isinstance(tail_ignore, int) and n is not None:
        return tail_ignore / (2 * n)
    elif isinstance(tail_ignore, float):
        return tail_ignore / 2
    else:
        return 0.0


def _downsample_dist_per_age(labnorm_df, quant_step=0.02, upper_quant_step=0.025, ages_to_consider=None,
                             ignore_tail: Union[int, float] = 50) -> dict:
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
        indices = [0]
        last_value = quant[0]
        for i in range(1, len(quant)):
            if quant[i] - last_value < quant_step:
                continue
            elif quant[i] - last_value <= upper_quant_step:
                indices.append(i)
                last_value = quant[i]
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


def _iter_df_age_quant_vals(df, groupby_gender=False):
    if groupby_gender:
        groupby = ["age", "gender"]
    else:
        groupby = ["age"]
    for group, group_df in df.groupby(groupby):
        group = list(group)  # tuple is immutable
        group[0] = _parse_age_labnorm(group[0])  # age is the first element
        n = int(group_df["n"].iloc[0]) if "n" in group_df.columns else None
        yield group, group_df["quant"], group_df["value"], n
