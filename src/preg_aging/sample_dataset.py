from .custom_dist import LinearInterpDistribution
from .labnorm_utils import get_unique_reference_ages_from_labnorm, age_to_quant_and_vals
import numpy as np

import pandas as pd


class AgeLabSampler:
    def __init__(self, lab_name: str, age: int, quants: pd.Series, vals: pd.Series):
        self.lab_name = lab_name
        self.ages = age
        self.sampler = LinearInterpDistribution(vals, quants)

    def sample(self, num_samples: int):
        return self.sampler.rvs(size=num_samples)


def resample_per_age(lab_names: list[str], num_samples_per_age=20, min_quant_step=0.02, tail_ignore=0.01, range_to_fac: dict[tuple[int, int], int] = None) -> (pd.DataFrame, pd.Series):
    """
    Resample the lab values per age. Order of samples is not guaranteed to be random.
    :param lab_names: list of lab names
    :param min_quant_step: minimum step for the quantile values:
    :param num_samples_per_age
    :return: dict of age to lab values
    """
    ages = get_unique_reference_ages_from_labnorm(lab_names)
    if range_to_fac is not None:
        labels = []
        for age_range, fac in range_to_fac.items():
            intersected_ages = np.intersect1d(ages, np.arange(*age_range))
            labels.extend([[age] * num_samples_per_age * fac for age in intersected_ages])
        labels = np.hstack(labels)
    else:
        labels = np.hstack([[age] * num_samples_per_age for age in ages])
    regression_table = np.empty((labels.size, len(lab_names)), dtype=float)
    for j, lab_name in enumerate(lab_names):
        quant_val_df = age_to_quant_and_vals(lab_name, ages=ages, min_quant_step=min_quant_step, ignore_tail=tail_ignore)
        for age in ages:
            quant = quant_val_df[age]["quant"]
            val = quant_val_df[age]["value"]
            row_indices = np.argwhere(labels == age).squeeze()
            lab_sample = AgeLabSampler(lab_name, age, quant, val).sample(
                row_indices.size)
            regression_table[row_indices, j] = lab_sample
    return pd.DataFrame(regression_table, columns=lab_names), pd.Series(labels, name="age")


def normalize_by_sub_div(regression_table):
    """

    :param regression_table: Shape is samples * labs + 1 (n * m)
    :return:
    """
    return (regression_table - regression_table.mean(axis=0)) / regression_table.std(axis=0)
