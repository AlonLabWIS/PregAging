from typing import Union

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, false_discovery_control
import statsmodels.api as sm

from .clalit_parser import get_clalit_data, get_mean_age_pre_conception, recalculate_mean_sd_bad_tests
from .labnorm_utils import find_median_for_lab, interp_per_age


def find_closet_age_median_value(test_name):
    df = get_clalit_data(test_name)
    median_reference = _read_labnorm_median(test_name)
    if median_reference.empty:
        return median_reference
    median_value = df["val_50"]
    week_to_age = {}
    for gest_week, median_gestational_value in median_value.items():
        closest_median = (median_reference - median_gestational_value).abs()
        closest_age = closest_median.idxmin()[0]  # Find the youngest age which minimizes the difference
        week_to_age[gest_week] = min(closest_age, 85)
    return pd.Series(week_to_age, index=week_to_age.keys(), name=test_name)


def model_labnorm_linreg(test_name, ages=(20, 80)):
    labnorm_median = _read_labnorm_median(test_name)
    if labnorm_median.empty:
        return
    y_model_vals = np.array(ages)
    x_model_vals = labnorm_median[y_model_vals]
    model = LinearRegression().fit(x_model_vals.to_numpy().reshape(-1, 1), y_model_vals)
    return model, (x_model_vals.values, y_model_vals)


def model_preconception_linreg(test_name, max_week=-58):
    pre_conception_ref = get_mean_age_pre_conception(test_name, max_week=max_week)["val_50"]
    x_model_vals = pre_conception_ref.to_numpy().reshape(-1, 1)
    y_model_vals = pre_conception_ref.index  # age group
    model = LinearRegression().fit(x_model_vals, y_model_vals)
    return model, (x_model_vals, y_model_vals)


def _read_labnorm_median(test_name):
    try:
        return find_median_for_lab(test_name)
    except FileNotFoundError:
        return pd.Series(name=test_name)


def reverse_1d_linear_model(model):
    if model.coef_ == 0:
        return None
    slope = 1 / model.coef_
    intercept = -model.intercept_ / model.coef_

    def inner(x: np.ndarray):
        return slope * np.array(x) + intercept

    return inner


def predict_pregnancy_age_per_test(test_name, preconception_predictor=None, labnorm_predictor=None):
    median_test_series = get_clalit_data(test_name)["val_50"]
    if preconception_predictor is None:
        preconception_predictor, _ = model_preconception_linreg(test_name)
    if labnorm_predictor is None:
        labnorm_predictor, _ = model_labnorm_linreg(test_name)
    labnorm_predictions = labnorm_predictor.predict(median_test_series.to_numpy().reshape(-1, 1))
    if preconception_predictor is True:
        preg_predictions = preconception_predictor.predict(median_test_series.to_numpy().reshape(-1, 1))
        return pd.DataFrame({"Pre-conception": preg_predictions, "LabNorm": labnorm_predictions},
                            median_test_series.index)
    else:
        return pd.Series(labnorm_predictions, median_test_series.index)


def find_pregnancy_amplitude(test_name, test_period, sample_size=100):
    df = get_clalit_data(test_name)
    test_qmean = df["qval_mean"].loc[test_period[0]:test_period[1]]
    test_period_qmeans = test_qmean.loc[test_period[0]:test_period[1]]
    test_n = df["val_n"].loc[test_period[0]:test_period[1]]
    test_qsd = df["qval_sd"].loc[test_period[0]:test_period[1]]
    is_positive = np.where(
        test_period_qmeans.max() - test_qmean.iloc[0] > test_period_qmeans.iloc[0] - test_period_qmeans.min(), 1, -1)
    # is_pos = 1 if test_period_qmeans.max() - test_qmean.iloc[0] > test_period_qmeans.iloc[0] - test_period_qmeans.min() else -1
    random_series = np.random.normal(test_qmean, test_qsd / (test_n ** 0.5),
                                     size=(sample_size, len(test_qmean)))  # shape is sample_size * weeks
    amps = (random_series.max(axis=1) - random_series.min(axis=1)) * is_positive
    return amps.mean(), amps.std()
    # preg_max = test_period_qmeans.max()
    # preg_min = test_period_qmeans.min()
    # return (preg_max.max() - preg_min.min()) * is_pos


def find_labnorm_amplitude(labnorm_age_ref, test_name, old_neighborhood=5):
    median_ref_vals_old = remove_linear_trend_labnorm(test_name, labnorm_age_ref[1], old_neighborhood)
    labnorm_ref_quant_old_at_young = interp_per_age(test_name, is_quantile=False)[(labnorm_age_ref[0],)](
        median_ref_vals_old)  # For the younger age, look quantile for old value
    diff_labnrom = labnorm_ref_quant_old_at_young.mean() - 0.5  # quantile of median at old age for the young age distribution. 0.5 because the young value is just the median.
    return diff_labnrom, labnorm_ref_quant_old_at_young.std()


def remove_linear_trend_labnorm(test_name, reference_age, neighborhood: Union[int, tuple[int, int]] = 5):
    labnorm_ref = find_median_for_lab(test_name)
    if isinstance(neighborhood, int):
        age_range = np.arange(reference_age - neighborhood, reference_age + neighborhood)
    else:
        age_range = np.arange(*neighborhood)
    labnorm_ref_age = labnorm_ref[age_range]
    ols_model = sm.OLS(labnorm_ref_age, sm.add_constant(labnorm_ref_age.index)).fit()
    if ols_model.pvalues.iloc[1] > 0.05:
        return labnorm_ref_age
    return ols_model.resid + labnorm_ref[
        reference_age].item()  # Really? rationale: Noise plus interecept, neutralize change in age


def sample_test_mean_val(test_name, num_samples, clalit_path=None, is_quantile=False, na_interpolate=True):
    if clalit_path is None:
        test_df = get_clalit_data(test_name)
    else:
        test_df = get_clalit_data(test_name, clalit_path)
    mean_col = test_df["qval_mean"] if is_quantile else test_df["val_mean"]
    sd_col = test_df["qval_sd"] if is_quantile else test_df["val_sd"]
    n_col = test_df["val_n"]
    if sd_col.isna().any() and na_interpolate:  # Then all columns need interpolation
        mean_col = mean_col.interpolate().bfill().ffill()
        sd_col = sd_col.interpolate().bfill().ffill()
        n_col = n_col.interpolate().bfill().ffill()
    sim = np.random.normal(mean_col, sd_col / n_col ** 0.5, size=(num_samples, len(test_df)))
    return pd.DataFrame(sim.T, index=test_df.index).stack()


def sample_tests_mean_val(tests, num_samples, clalit_path=None, is_quantile=False, na_interpolate=True):
    simulation_res = None
    for test_name in tests:
        ser = sample_test_mean_val(test_name, num_samples, clalit_path, is_quantile, na_interpolate)
        if simulation_res is None:
            simulation_res = pd.DataFrame(ser.to_numpy(), columns=[test_name], index=ser.index)
        else:
            simulation_res[test_name] = ser
    return simulation_res


def sample_regression_params(model_param_covar, model_param_expectation, num_samples):
    return np.random.multivariate_normal(model_param_expectation, model_param_covar, num_samples)


def z_test_spinoff_two_sided(values, compared_val=0, alternative="two-sided"):
    z_scores = (values.mean(axis=0) - compared_val) / values.std(axis=0)
    if alternative == "two-sided":
        p_values = 2 * norm.cdf(-np.abs(z_scores))  # two-sided z-test
    elif alternative == "less":
        p_values = norm.cdf(z_scores)
    elif alternative == "greater":
        p_values = 1 - norm.cdf(z_scores)
    else:
        raise ValueError(f"Alternative can be 'two-sided', 'less' or 'greater', got '{alternative}'")
    return _na_false_discovery_control(p_values)


def count_zero_crossing_symmetric(data, alternative="two-sided"):
    if alternative == "two-sided":
        return (np.sign(data).diff().abs() / 2).sum(axis=0)
        diff_pos_neg = np.sign(data).sum(axis=0).abs()
    elif alternative == "less":
        diff_pos_neg = (data < 0).sum(axis=0)
    elif alternative == "greater":
        diff_pos_neg = (data > 0).sum(axis=0)
    p_vals = 1 - diff_pos_neg / data.shape[0]
    return _na_false_discovery_control(np.clip(p_vals, 1 / data.shape[0], None))  # Cannot have 0 p-value


def scr_to_egfr(scr, age):
    return 142 * np.min(scr / 0.7, 1) ** (-0.241) * np.max(scr / 0.7, 1) ** (-1.2) * 0.9938 ** age


def _na_false_discovery_control(p_values):
    corrected = np.empty(p_values.shape)
    na_inds = np.isnan(p_values)
    corrected[~na_inds] = false_discovery_control(p_values[~na_inds])
    corrected[na_inds] = np.nan
    return corrected
