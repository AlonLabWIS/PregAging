from typing import Union, Sequence, Iterable

import pandas as pd
import numpy as np
from patsy.eval import test_ast_names
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, false_discovery_control
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm

from .clalit_parser import get_clalit_data, get_mean_age_pre_conception, get_quantiles_from_column_names
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


def identify_outliers(test_df, v95_v5_to_sd_ratio=10):
    return ((test_df["val_95"] - test_df["val_5"]) * v95_v5_to_sd_ratio) < test_df["val_sd"]


def linearly_fix_outliers(test_df, v95_v5_to_sd_ratio=10):
    outliers_indices = identify_outliers(test_df, v95_v5_to_sd_ratio)
    if not outliers_indices.any():
        return None
    test_df_c = test_df.copy()
    quantiles, columns = get_quantiles_from_column_names(test_df_c.columns)
    sorted_quantiles = np.argsort(quantiles)
    columns = np.array(columns)[sorted_quantiles]
    quantiles = np.array(quantiles)[sorted_quantiles]
    mean_quant, sd_quant = recalculate_mean_sd_lin_approx(quantiles, test_df_c.loc[outliers_indices, columns].values)
    test_df_c.loc[outliers_indices, ["val_mean", "val_sd"]] = np.stack([mean_quant, sd_quant], axis=1)
    return test_df_c


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


def predict_age_in_pregnancy(model: lm.RegressionResults, clalit_path: Union[None, str] = None, sample_size: int = 1000,
                             exclude_points: Union[None, Iterable[float]] = None, fix_outliers: bool = True,
                             compared_path: Union[None, str] = None) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Generate a sample per each weekly bin in the dataset based on the mean and standard deviation.
    For each such sample, generate parameters from the regression model based on their value and standard deviation.
    Predict each such sample and return the raw results and aggregated by mean and standard deviation per weekly bin.
    :param model: The regression model. The parameters of the model have to be named (by index) as valid lab test file names in the Clalit directory.
    :param clalit_path: The path to directory with valid lab test CSV files. If None, the default path is used.
    :param compared_path: Path to a different Clalit directory. If given, the prediction is on the difference `compared-clalit`.
    :param sample_size: The number of samples to generate per weekly bin.
    :param exclude_points: Weekly bins to drop. If None, drops none. Otherwise, drops every weekly bin excluded. Values are in `Week postpartum` unit.
    :param fix_outliers: Fix outliers in the Clalit data. An outlier weekly bin has a difference between the 95,5 percentiles much smaller than the standard deviation.
    Mean and standard deviation are re-calculated using linear interpolation of the weekly bin's percentile values to approximate a CDF.
    :return: A 2-tuple with the first item a DataFrame with column "age" the mean prediction in the weekly bin and "sd" the standard deviation of the prediction. The index is the weekly bin for all tests.
    The second an unaggregated series named "age" with multiindex. Top level named "week" and second level unnamed.
    """
    test_names = model.params.index[model.params.index != "const"]
    if clalit_path is None:
        sampled_preg = sm.add_constant(sample_tests_mean_val(test_names, sample_size, fix_outliers=fix_outliers))
    else:
        sampled_preg = sm.add_constant(sample_tests_mean_val(test_names, sample_size, clalit_path, fix_outliers=fix_outliers))
    if compared_path:
        sampled_preg = sm.add_constant(sample_tests_mean_val(test_names, sample_size, compared_path, fix_outliers=fix_outliers)) - sampled_preg
    params = sample_regression_params(model.cov_params(), model.params, sampled_preg.shape[0])
    # Inner product of each pair of rows in the two matrices.
    unagg_pred = pd.Series(data=np.einsum('ij,ij->i', sampled_preg, params), index=sampled_preg.index,
                           name="age")
    if exclude_points is not None:
        unagg_pred = unagg_pred[~unagg_pred.index.isin(exclude_points, 0)]
    pred = unagg_pred.groupby(level=0).agg(age="mean", sd="std")
    return pred, unagg_pred


def find_pregnancy_amplitude(test_name, test_period, sample_size=100, clalit_path=None):
    simulated_series = simulate_pregnancy_quantile_data(test_name, test_period, sample_size, clalit_path)
    mean_series = simulated_series.mean(axis=0)
    is_positive = np.where(
        mean_series.max() - mean_series[0] > mean_series[0] - mean_series.min(), 1, -1)
    amps = (simulated_series.max(axis=1) - simulated_series.min(axis=1)) * is_positive
    return amps.mean(), amps.std()


def find_diff_amplitude(test_name: str, other_clalit_path: str, test_period: tuple[float, float], sample_size: int = 100,
                        clalit_path_base: Union[None, str] = None) -> tuple[float, float]:
    """
    Calculate the maximal the amplitude of the difference in mean quantile score between "other" and "base" clalit data.
    :param test_name: Lab test name (valid CSV file name).
    :param other_clalit_path: Path to directory with CSV files of lab tests.
    :param test_period: The period of time to take the data from in "week postpartum" unit. Gestation is [-40., 0.]
    :param sample_size: How many times to repeat the simulation (to generate the mean and standard deviation)
    :param clalit_path_base: The base path to the clalit data. If None, the default path is used.
    Resulting simulation values is subtracted from the "other" simulation values by week postpartum.
    :return: A 2-tuple with the first value the mean and the second value the standard deviation of the maximal difference.
    """
    base_sim = simulate_pregnancy_quantile_data(test_name, test_period, sample_size, clalit_path_base)
    new_sim = simulate_pregnancy_quantile_data(test_name, test_period, sample_size, other_clalit_path)
    path_difference = new_sim - base_sim
    path_diff_arg_max = np.abs(path_difference.mean(axis=0)).argmax()
    # For all repeats, choose the week in the period with the largest difference
    path_diff_at_max = path_difference[:, path_diff_arg_max]
    return path_diff_at_max.mean().item(), path_diff_at_max.std().item()


def simulate_pregnancy_quantile_data(test_name: str, test_period: tuple[float, float], sample_size: int = 100,
                                     clalit_path: Union[None, str] = None):
    """
    Create a sample from normal distribution based on the mean and standard deviation of the quantile data for a lab test per week.
    :param test_name: Name of the lab test. Must be a valid CSV file name.
    :param test_period: The time period to consider for the simulation in units of "week postpartum". Data outside the range is not considered.
    First item is the lower bound on the gestaional week and the second is the upper bound.
    :param sample_size: Number of repeats for the simulation.
    :param clalit_path: Path to Clalit direcotry with valid CSV files. If None, the default path is used.
    :return: a float numpy array of shape (sample_size, weeks_in_range) with the simulated values.
    """
    if clalit_path is None:
        df = get_clalit_data(test_name)
    else:
        df = get_clalit_data(test_name, clalit_path)
    test_qmean = df["qval_mean"].loc[(slice(*test_period), ...), :]
    test_n = df["val_n"].loc[(slice(*test_period), ...), :]
    test_qsd = df["qval_sd"].loc[(slice(*test_period), ...), :]
    return np.random.normal(test_qmean, test_qsd / (test_n ** 0.5),
                            size=(sample_size, len(test_qmean)))  # shape is sample_size * weeks


def find_labnorm_amplitude(labnorm_age_ref: tuple[float, float], test_name: str, old_neighborhood: int = 5) -> tuple[
    float, float]:
    """
    Find the change in age between median values. The median value at old age is compared (with a neighborhood around the old age) against the quantile in the young age.
    The mean and standard deviation of the difference is returned. Any change in the neighborhood is removed, assuming a linear effect with age.
    :param labnorm_age_ref: The young age for the reference is the first value and the old age is the second value.
    :param test_name: Lab test name, a valid CSV LabNorm file.
    :param old_neighborhood: The radius of the neighborhood for the old age. Larger radius gives a longer series for the mean and standard deviation but may introduce aging effect.
    :return: The mean (first item) and the standard deviation (second item) of the quantile difference in age.
    """
    median_ref_vals_old = remove_linear_trend_labnorm(test_name, labnorm_age_ref[1], old_neighborhood)
    labnorm_ref_quant_old_at_young = interp_per_age(test_name, is_quantile=False)[(labnorm_age_ref[0],)](
        median_ref_vals_old)  # For the younger age, look quantile for old value
    diff_labnrom = labnorm_ref_quant_old_at_young.mean() - 0.5  # quantile of median at old age for the young age distribution. 0.5 because the young value is just the median.
    return diff_labnrom.item(), labnorm_ref_quant_old_at_young.std().item()


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


def sample_test_mean_val(test_name, num_samples, clalit_path=None, is_quantile=False, na_interpolate=False,
                         fix_outliers=False):
    if clalit_path is None:
        test_df = get_clalit_data(test_name)
    else:
        test_df = get_clalit_data(test_name, clalit_path)
    if fix_outliers and not is_quantile:
        fixed_df = linearly_fix_outliers(test_df)
        test_df = fixed_df if fixed_df is not None else test_df
    mean_col = test_df["qval_mean"] if is_quantile else test_df["val_mean"]
    sd_col = test_df["qval_sd"] if is_quantile else test_df["val_sd"]
    n_col = test_df["val_n"]
    if sd_col.isna().any() and na_interpolate:  # Then all columns need interpolation
        mean_col = mean_col.interpolate().bfill().ffill()
        sd_col = sd_col.interpolate().bfill().ffill()
        n_col = n_col.interpolate().bfill().ffill()
    sim = np.random.normal(mean_col, sd_col / n_col ** 0.5, size=(num_samples, len(test_df)))
    return pd.DataFrame(sim.T, index=test_df.index).stack()


def sample_tests_mean_val(tests, num_samples, clalit_path=None, is_quantile=False, na_interpolate=True,
                          fix_outliers=False):
    simulation_res = None
    for test_name in tests:
        ser = sample_test_mean_val(test_name, num_samples, clalit_path, is_quantile, na_interpolate, fix_outliers)
        if simulation_res is None:  # First test
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
    return p_values if np.isscalar(p_values) else _na_false_discovery_control(p_values)


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
    return 142 * (np.min([scr / 0.7, 1]) ** (-0.241) * np.max([scr / 0.7, 1]) ** (-1.2)) * (0.9938 ** age)


def scr_to_egfr_preg(scr, gw):
    #  88.4 is g/mL to mmol/L conversion. This is the paper: https://www.nature.com/articles/s41598-024-57737-0
    return (2 - scr * 88.4 / 55.25) * 103.1 * 55.25 / (
            56.7 - 0.223 * gw - 0.113 * gw ** 2 + 0.00545 * gw ** 3 - 0.0000653 * gw ** 4)


def calculate_model_weights_ci(model: lm.RegressionResults, alpha: float = 0.05) -> pd.DataFrame:
    """
    Organize the regression model's weights without intercept and confidence intervals into a DataFrame.
    :param model: The regression model results
    :param alpha: The confidence level is 1-alpha bilateral.
    :return: A dataframe with index as the feature names. "value" column is the weight, "bottom_ci" and "top_ci" are the confidence interval of 1-alpha/2 of either side.
    """
    names_to_weights = model.conf_int(alpha).iloc[1:].rename(columns={0: "bottom_ci", 1: "top_ci"})  # skip intercept
    names_to_weights["value"] = model.params.iloc[1:]  # skip intercept
    return names_to_weights


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


# TODO is this function really necessary? seems awfully similar to the one below
def _find_target_linear(x1, x2, y1, y2, target):
    dx = (x2 - x1)
    bad_indices = dx == 0
    dx[bad_indices] = 1  # Avoid division by zero
    slope = (y2 - y1) / dx
    slope[bad_indices] = np.nan  # Remove faulty slopes
    return (target - y1) / slope + x1


def _find_slopes_pieces(xs: np.ndarray, ys: np.ndarray):
    dx = xs[:, 1:] - xs[:, :-1]
    bad_indices = dx == 0
    dx[bad_indices] = 1  # Avoid division by zero
    slopes = (ys[1:] - ys[:-1]) / dx
    slopes[bad_indices] = np.nan  # Remove faulty slopes
    return np.pad(slopes, ((0, 0), (1, 1)), mode="edge")


def _na_false_discovery_control(p_values):
    corrected = np.empty(p_values.shape)
    na_inds = np.isnan(p_values)
    corrected[~na_inds] = false_discovery_control(p_values[~na_inds])
    corrected[na_inds] = np.nan
    return corrected
