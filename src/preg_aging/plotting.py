from typing import Iterable, Sequence
import os
import json

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr, false_discovery_control
# from adjustText import adjust_text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tools import add_constant
from preg_aging.analyses import z_test_spinoff_two_sided, count_zero_crossing_symmetric

from .analyses import predict_pregnancy_age_per_test, model_labnorm_linreg, reverse_1d_linear_model, \
    find_pregnancy_amplitude, find_labnorm_amplitude, sample_tests_mean_val, sample_regression_params
from .clalit_parser import get_clalit_data, translate_long_to_short_lab, get_data_by_tests_and_field, \
    get_condition_from_filename
from .labnorm_utils import find_median_for_lab, get_mean_std_per_test

_COLORS = ["red", "blue", "green", "orange", "brown"]


def remove_top_right_frame(axes: Iterable[Axes]):
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def remove_ax_labels(ax: Axes, remove_x: bool = False, remove_y: bool = False):
    ax.set_title('')
    if remove_x:
        ax.set_xlabel('')
        ax.set_xticks([])
    if remove_y:
        ax.set_yticks([])
        ax.set_ylabel('')


def finalize_weeks_postpartum_plots(ax: plt.Axes):
    ax.set_xlabel("Week postpartum", fontsize=14)
    ax.axvspan(-38, 0, color='dimgray', alpha=0.1, zorder=-20, label="Pregnancy")
    ax.grid(False)
    ax.set_xlim([-60, 80])
    remove_top_right_frame([ax])


def finalize_preg_panel(ax: Axes, span_label, xverts=()):
    for xvert in xverts:
        ax.axvline(x=xvert, color='k', linestyle='--', lw=0.2, alpha=0.1)
    if xverts:
        ax.set_xticks(xverts)
    else:
        ax.set_xticks(np.arange(-50, 100, 50))
    finalize_weeks_postpartum_plots(ax)

    ax.tick_params(axis='both', which='major', labelsize=6)


def plot_series_seq(series_seq: Sequence[pd.Series], ncols=6, ylabel=None):
    axes, fig, n = _grid_by_seq(ncols, series_seq)
    for i, (ax, series) in enumerate(zip(axes.flat, series_seq)):
        if not series.empty:
            ax.plot(series.index, series.values, color="k", lw=0.5)
            finalize_preg_panel(ax, "Pregnancy")
            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=10)
        ax_name = translate_long_to_short_lab([series.name]).iloc[0]
        ax.annotate(ax_name, (0.05, 1.05), xycoords='axes fraction', fontsize=10, color='k')
        remove_x_labels = i < (n - ncols)
        remove_ax_labels(ax, remove_x=remove_x_labels, remove_y=series.empty)
    plt.tight_layout()
    return fig, axes


def _grid_by_seq(ncols, series_seq, panel_size=2):
    n = len(series_seq)
    nrows = (n - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size, nrows * panel_size))
    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis('off')
    return axes, fig, n


def plot_df_seq(group_name: str, test_names: Sequence[str], df_seq: Sequence[pd.Series], ncols=6, ylabel="age"):
    axes, fig, n = _grid_by_seq(ncols, df_seq)
    for i, (ax, df) in enumerate(zip(axes.flat, df_seq)):
        remove_x_labels = i < (n - ncols)
        plot_single_test_df(ax, df, test_names[i], ylabel, remove_x_labels)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"System: {group_name}", fontsize=16, y=1.0)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles=handles[:3], labels=labels[:3], loc="lower right")
    plt.tight_layout()
    return fig, axes


def plot_single_test_df(ax, pd_obj, test_name, ylabel, remove_x_labels=False, annotation_coords=(0.05, 1.05),
                        ha="right"):
    if isinstance(pd_obj, pd.Series):
        ax.plot(pd_obj.index, pd_obj.values, color='r', lw=0.5)
    else:
        for j, col in enumerate(pd_obj.columns):
            ax.plot(pd_obj.index, pd_obj[col].values, color=_COLORS[j], lw=0.5, label=col)
    finalize_preg_panel(ax, "Pregnancy", (-38, -25, 0, 10))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    ax_name = translate_long_to_short_lab([test_name]).iloc[0]
    if annotation_coords is None:
        annotation_coords = (-0.05, 1.08)
    ax.annotate(ax_name, annotation_coords, xycoords='axes fraction', fontsize=10, color='k', ha=ha)
    remove_ax_labels(ax, remove_x=remove_x_labels)


def plot_linear(ax, line_x_points, rev_model, x_ticks=None, extra_points=None, xlabel="Age",
                ylabel="test units", c="k"):
    if rev_model is not None:
        line_y_points = rev_model(line_x_points)
        ax.plot(line_x_points, line_y_points, color=c, lw=0.5)
        if extra_points is not None:
            ax.scatter(extra_points[0], extra_points[1], c=c, s=10)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor('w')
    remove_top_right_frame([ax])


def plot_tests_labnorm_age_on_ax_color_mapper(test_names: Sequence[str], ax, colormap=None, normalize=False,
                                              color=None):
    norm = Normalize(vmin=0, vmax=len(test_names) - 1, clip=True)
    if colormap is None:
        color_mapper = cm.ScalarMappable(norm=norm, cmap="turbo")
    else:
        color_mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
    pd_series = []
    for test_name in test_names:
        labnorm_predict = predict_pregnancy_age_per_test(test_name, preconception_predictor=False)
        if normalize:
            labnorm_predict = labnorm_predict / labnorm_predict.abs().max()
        labnorm_predict.name = translate_long_to_short_lab([test_name]).item()
        pd_series.append(labnorm_predict)
    df = pd.concat(pd_series)
    if color is None:
        for i, test_name in enumerate(test_names):
            ax.plot(labnorm_predict.index, labnorm_predict.values, c=color, lw=0.5, alpha=0.5)
            coord = np.abs(labnorm_predict.values).argmax()
            ax.text(labnorm_predict.index[coord], labnorm_predict.values[coord],
                    translate_long_to_short_lab([test_name]).item(), ha='center', fontsize=10, color=color)
    else:
        plot_color_mapper_on_axes(ax, df, color_mapper=color_mapper)
    finalize_preg_panel(ax, "Pregnancy")
    return


def plot_color_mapper_on_axes(ax, df, color_mapper="turbo"):
    test_names = df.columns
    test_names_annotate = translate_long_to_short_lab(test_names).values
    norm = Normalize(vmin=0, vmax=len(test_names) - 1, clip=True)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=color_mapper)
    for i, test_name in enumerate(test_names.values):
        ser = df[test_name]
        coord = np.abs(ser).argmax()
        c = color_mapper.to_rgba(np.array(i))
        ax.text(ser.index[coord], ser.values[coord], test_names_annotate[i], ha='center', fontsize=10, color=c)
        ax.plot(df.index, ser, c=c, lw=0.5, alpha=0.5)
    return color_mapper


def plot_tests_trends(json_path=os.path.join("csvs", "tests_trends.json"), normalize=True):
    with open(json_path, "r") as f:
        trends = json.load(f)
    ncols = 2
    nrows = len(trends)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.9, nrows * 3))
    for i, (group_name, trend_tests) in enumerate(trends.items()):
        for j, (trend, sig_tests) in enumerate(trend_tests.items()):
            ax = axes[i][j]
            for sig_test in sig_tests:
                if sig_test["is_sig"]:
                    plot_tests_labnorm_age_on_ax_color_mapper(sig_test["tests"], ax, normalize=normalize)
                else:
                    plot_tests_labnorm_age_on_ax_color_mapper(sig_test["tests"], ax, normalize=True,
                                                              color="gray")
        axes[i][0].annotate(group_name + ":", (0.05, 1.05), xycoords='axes fraction', fontsize=12, color='k',
                            ha="right")

    fig.suptitle("Trends", fontsize=16, y=1.0)
    fig.subplots_adjust(wspace=0.1, hspace=1.8)
    plt.tight_layout()
    return fig, axes


def plot_diffs_histogram(test_names, test_period=[-40, 0], labnorm_age_ref=[20, 80], bins=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-1, 1, 2 * (bins - 2))
    preg_diffs = []
    labnorm_diffs = []
    for test_name in test_names:
        diff_labnrom, _ = find_labnorm_amplitude(labnorm_age_ref, test_name)
        labnorm_diffs.append(diff_labnrom)
        pregnancy_diff_mean, _ = find_pregnancy_amplitude(test_name, test_period)
        preg_diffs.append(pregnancy_diff_mean)

    ax.hist([preg_diffs, labnorm_diffs], bins=bins, label=['Gestation', "Aging"], color=['k', 'grey'])
    ax.grid(False)
    ax.set_xlabel("Quantile change", fontsize=14)
    ax.set_ylabel("#Tests", fontsize=14)
    remove_top_right_frame([ax])
    ax.legend(loc="upper right", fontsize=12)
    return fig, ax


def plot_quantile_diffs_pregnancy_ref(ax, test_names, labnorm_age_ref=[20, 80], test_period=[-40, -1]):
    for test_name in test_names:
        diff_labnorm, diff_labnorm_sd = find_labnorm_amplitude(labnorm_age_ref, test_name)
        pregnancy_diff_mean, pregnancy_diff_sd = find_pregnancy_amplitude(test_name, test_period)
        ax.errorbar(diff_labnorm, pregnancy_diff_mean, xerr=2 * diff_labnorm_sd, yerr=2 * pregnancy_diff_sd, c="k",
                    marker="o", markersize=5,
                    capsize=2)
        ax.annotate(translate_long_to_short_lab([test_name]).item(), (diff_labnorm, pregnancy_diff_mean), fontsize=13,
                    textcoords="offset points", xytext=(5, 5), ha='center')
    # adjust_text(texts, ax=ax)
    style_quartile_plots(ax, test_period, f"Reference quantile difference")


def plot_diff_pathologies(ax, test_names, pathology_path, healthy_path, test_period=[-40, 0], labnorm_age_ref=[20, 80],
                          c="k"):
    for test_name in test_names:
        pathology = get_clalit_data(test_name, pathology_path).loc[test_period[0]:test_period[1], "qval_mean"]
        healthy = get_clalit_data(test_name, healthy_path).loc[test_period[0]:test_period[1], "qval_mean"]
        diff_labnrom = find_labnorm_amplitude(labnorm_age_ref, test_name)
        path_diff = pathology - healthy
        path_diff_max = path_diff.iloc[path_diff.abs().argmax()]
        ax.scatter(diff_labnrom, path_diff_max, marker="o", s=10, c=c, label="Pre-eclampsia")
        ax.annotate(translate_long_to_short_lab([test_name]).item(), (diff_labnrom, path_diff_max), fontsize=13,
                    textcoords="offset points", xytext=(5, 5), ha='center')
    style_quartile_plots(ax, test_period, f"Labnorm quantile diff (age, [{labnorm_age_ref[0]}, {labnorm_age_ref[1]}])")


def plot_quantile_diff_pregnancy_model_weights(ax, test_names, model_weights, test_period=[-40, 0]):
    for test_name in test_names:
        weight = model_weights.loc[test_name, "value"]
        bottom_ci = weight - model_weights.loc[test_name, "bottom_ci"]  # error bar values are positive
        top_ci = model_weights.loc[test_name, "top_ci"] - weight
        pregnancy_diff, pregnancy_sd = find_pregnancy_amplitude(test_name, test_period)
        ax.errorbar(weight, pregnancy_diff, xerr=np.array((bottom_ci, top_ci)).reshape(-1, 1), yerr=2 * pregnancy_sd,
                    c="k", markersize=5,
                    fmt="o")
        ax.annotate(translate_long_to_short_lab([test_name]).item(), (weight, pregnancy_diff), fontsize=13,
                    textcoords="offset points", xytext=(5, 5), ha='center')
    # adjust_text(texts, ax=ax)
    style_quartile_plots(ax, test_period, f"Linear model weight", False)


def style_quartile_plots(ax, test_period, xlabel, same_scale=True):
    if same_scale:
        max_xy = np.abs(ax.get_xlim() + ax.get_ylim()).max()  # plus sign is concatenation
        max_xy *= 1.05  # add margin
        ax.set_xlim(-max_xy, max_xy)
        ax.set_ylim(-max_xy, max_xy)
        ax.plot([-max_xy, max_xy], [-max_xy, max_xy], '--', c="gainsboro", lw=0.5)
        ax.plot([-max_xy, max_xy], [max_xy, -max_xy], '--', c="gainsboro", lw=0.5)
        max_x = max_xy
        max_y = max_xy
    else:
        max_x = np.abs(ax.get_xlim()).max() * 1.05
        max_y = np.abs(ax.get_ylim()).max() * 1.05
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)

    # 1st
    ax.fill_between([0, max_x], max_y, color='orangered', alpha=0.05)
    # 2nd
    ax.fill_between([-max_x, 0], max_y, color='lightgreen', alpha=0.05)
    # 3rd
    ax.fill_between([-max_x, 0], -max_y, color='orangered', alpha=0.05)
    # 4th
    ax.fill_between([0, max_x], -max_y, color='lightgreen', alpha=0.05)
    ax.annotate("aging", xy=(max_x / 2, max_y / 2), fontsize=12, color='k',
                ha='center')
    ax.annotate("rejuvenation", xy=(-max_x / 2, max_y / 2), fontsize=12,
                color='k', ha='center')
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Pregnancy max quantile difference")


def grid_test_groups(test_groups, ncols):
    nrows = (len(test_groups) - 1) // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    flattened_axs = axs.flatten()
    # Remove axes for any unused subplots
    test_groups_keys = list(test_groups.keys())
    for i in range(0, len(flattened_axs)):
        if i < len(test_groups):
            flattened_axs[i].annotate(test_groups_keys[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=14)
        else:
            flattened_axs[i].axis('off')
    return fig, axs


def plot_diff_grid(test_groups, labnorm_age_ref=[20, 80], test_period=[-40, 0], ncols=3):
    fig, axs = grid_test_groups(test_groups, ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        plot_quantile_diffs_pregnancy_ref(flattened_axs[i], test_groups[group], labnorm_age_ref, test_period)
    return fig, axs


def plot_diff_grid_pathologies(test_groups, pathology_path, healthy_path, labnorm_age_ref=[20, 80],
                               test_period=[-40, 0], ncols=3, color="k"):
    fig, axs = grid_test_groups(test_groups, ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        plot_diff_pathologies(flattened_axs[i], test_groups[group], pathology_path, healthy_path, test_period,
                              labnorm_age_ref, color)
    return fig, axs


def plot_diff_grid_model(test_groups, model_weights, test_period=[-40, 0], ncols=3):
    fig, axs = grid_test_groups(test_groups, ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        plot_quantile_diff_pregnancy_model_weights(flattened_axs[i], test_groups[group], model_weights, test_period)
    return fig, axs


def plot_groups_linear_prediction(test_groups, model_parameters_ser, preconception_period=[-60, -40], ncols=3,
                                  clalit_field="val_50"):
    fig, axs = grid_test_groups(test_groups, ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        tests_to_consider = test_groups[group]
        group_df = get_data_by_tests_and_field(tests_to_consider, clalit_field)
        considered_parameters = model_parameters_ser[tests_to_consider]
        prediction = group_df @ considered_parameters
        prediction -= prediction.loc[preconception_period[0]:preconception_period[1]].mean()
        individual_predictions = group_df * considered_parameters
        individual_predictions -= individual_predictions.loc[preconception_period[0]:preconception_period[1]].mean()
        plot_color_mapper_on_axes(flattened_axs[i], individual_predictions)
        flattened_axs[i].plot(prediction.index, prediction.values)
    return fig, axs


def plot_labnorm_age_to_mean(group_name: str, test_names: list[str], test_to_coefficient=None, ncols=7):
    axes, fig, n = _grid_by_seq(ncols, test_names, 3.5)
    flat_axes = axes.flatten()
    tests_stats = pd.DataFrame(columns=["test", "pearson-p", "spearman", "coef"])
    for i in range(n):
        ax = flat_axes[i]
        test_name = test_names[i]
        test_coef = test_to_coefficient[test_names[i]] if test_to_coefficient is not None else None
        labnorm_spearman_stat, labnrom_r_pvalue = plot_labnorm_mean_std(ax, test_name, test_coef)
        tests_stats.loc[i] = [test_name, labnrom_r_pvalue, labnorm_spearman_stat, test_coef]
    fig.suptitle(f"System: {group_name}", fontsize=16, y=1.3)
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    return fig, tests_stats.set_index("test")


def plot_labnorm_mean_std(ax, test_name, test_coef=None):
    df = get_mean_std_per_test(test_name, smoothing_width=7)
    mean = df['mean'].values
    labnrom_r_pvalue = pearsonr(df.index[~np.isnan(mean)], mean[~np.isnan(mean)]).pvalue
    labnorm_spearman_stat = spearmanr(df.index[~np.isnan(mean)], mean[~np.isnan(mean)]).statistic
    ax.plot(df['mean'], df.index, label='Mean', color='red')
    ax.fill_betweenx(df.index, df['mean'] - df['std'], df['mean'] + df['std'], color='blue', alpha=0.2, label='Std Dev')
    # ax.plot(df.index, df['mean'], label='Mean', color='red')
    # ax.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], color='blue', alpha=0.2, label='Std Dev')
    ax.set_title(translate_long_to_short_lab([test_name]).item())
    if test_coef is not None:
        stats = f"Pear. p-val: {labnrom_r_pvalue:.2f}\nSpearman: {labnorm_spearman_stat:.2f}\nCoef.: {test_coef:.2f}"
    else:
        stats = f"Pear. p-val: {labnrom_r_pvalue:.2f}\nSpearman: {labnorm_spearman_stat:.2f}"
    ax.annotate(stats, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10, ha='right', va='top')
    return labnorm_spearman_stat, labnrom_r_pvalue


def plot_test_with_linear_models(group_name, test_names: Sequence[str]):
    ncols = 3
    nrows = len(test_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2))
    for i, test_name in enumerate(test_names):
        labnorm_model, (labnorm_val_train, labnorm_age_train) = model_labnorm_linreg(test_name)
        age_predictions_df = predict_pregnancy_age_per_test(test_name, labnorm_predictor=labnorm_model,
                                                            preconception_predictor=False)
        median_test_values = get_clalit_data(test_name)["val_50"]
        edge_ages = [15, 61]
        age_ticks = np.arange(edge_ages[0], edge_ages[1], 15)
        reverse_labnorm_model = reverse_1d_linear_model(labnorm_model)
        labnorm_age_to_median = find_median_for_lab(test_name)
        labnrom_r_pvalue = pearsonr(labnorm_age_to_median.index, labnorm_age_to_median.values).pvalue
        labnorm_spearman_stat = spearmanr(labnorm_age_to_median.index, labnorm_age_to_median.values).statistic
        for j in range(ncols):
            ax: plt.Axes = axes[i][j]
            if j == 0:
                plot_single_test_df(ax, age_predictions_df, test_name, "age")
                if i == 0:
                    ax.set_title("Predicted Age", fontsize=12, y=1.08)
            elif j == 1:
                ax.plot(median_test_values.index, median_test_values.values, color="k", lw=0.5)
                finalize_preg_panel(ax, "Pregnancy")
                if i == 0:
                    ax.set_title("Test Value", fontsize=12, y=1.08)
            elif j == 2:
                # Colors indexing is bad and likely cause confusion with each change
                plot_linear(ax, edge_ages, reverse_labnorm_model, x_ticks=age_ticks,
                            extra_points=(labnorm_age_train, labnorm_val_train), c=_COLORS[0])
                ax.plot(labnorm_age_to_median.index, labnorm_age_to_median.values, c=_COLORS[0], lw=0.5, alpha=0.3)
                ax.annotate(f"Pear. p-val: {labnrom_r_pvalue:.2f}\nSpearman: {labnorm_spearman_stat:.2f}", (0.05, 0.95),
                            va="bottom",
                            xycoords='axes fraction', fontsize=8, color='k')
                if i == 0:
                    ax.set_title("Labnorm Linear Model", fontsize=12, y=1.08)
    fig.suptitle(f"System: {group_name}", fontsize=16, y=1.0)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    return fig, axes


def plot_age_acceleration_by_lin_reg(model, relative_path, compared_paths, sample_size=1000, exclude_points=None,
                                     compared_color_map=None,
                                     p_val_bins=None, use_fill=False):
    test_names = model.params.index[model.params.index != "const"]
    reps = [sample_tests_mean_val(test_names, sample_size, compared_path) for compared_path in compared_paths]
    reference = sample_tests_mean_val(test_names, sample_size, relative_path)
    preds = []
    for i, rep in enumerate(reps):
        params = sample_regression_params(model.cov_params(), model.params, rep.shape[0])
        condition = get_condition_from_filename(compared_paths[i]) if compared_color_map is None else \
        tuple(compared_color_map.keys())[i]
        pred = pd.DataFrame(data=np.einsum('ij,ij->i', add_constant(rep) - add_constant(reference), params),
                            columns=["age"], index=rep.index)
        pred["condition"] = condition
        preds.append(pred)
    preds = pd.concat(preds)  # week, unnamed column are part of the dataframe now
    if exclude_points is not None:
        preds = preds[~preds.index.get_level_values("week").isin(exclude_points)]
    weekly_stats = preds.reset_index().groupby(["week", "condition"])["age"].agg(age="mean", sd="std").reset_index()
    conditions = weekly_stats["condition"].unique()
    # Figure preparation
    colors = [compared_color_map[condition] for condition in
              conditions] if compared_color_map is not None else plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    fig, axs = plt.subplots(len(conditions), 1, figsize=(10, 6.8 * len(conditions)))
    for i, (condition, color) in enumerate(zip(conditions, colors)):
        ax = axs[i]
        condition_data = weekly_stats[weekly_stats["condition"] == condition]
        if use_fill:
            ax.plot(condition_data["week"], condition_data["age"], '-', color=color, label=condition)
            ax.fill_between(condition_data["week"], condition_data["age"] - condition_data["sd"] * 1.96,
                            condition_data["age"] + condition_data["sd"] * 1.96, color=color, alpha=0.2)
        else:
            ax.errorbar(x=condition_data["week"], y=condition_data["age"], yerr=condition_data["sd"] * 1.96, fmt='o-',
                        color=color, label=condition)
        if p_val_bins is not None:
            x_locs = []
            p_vals = []
            last_ind_used = None
            for j in range(0, len(p_val_bins) - 1):
                if last_ind_used is None:
                    last_ind_used = p_val_bins[j]
                right = p_val_bins[j + 1]
                bin_vals = preds.loc[preds["condition"] == condition, "age"].loc[last_ind_used:right]
                last_ind_used = bin_vals.index.get_level_values(0).max() + 1
                p_vals.append(z_test_spinoff_two_sided(bin_vals.values, alternative="greater"))
                x_locs.append((p_val_bins[j] + right) / 2)
                if j != 0:  # don't add vertical line for the xmin value
                    ax.axvline(x=p_val_bins[j], color="gray", linestyle="--", lw=2, alpha=0.3)
            p_vals = false_discovery_control(p_vals)
        else:
            p_vals = z_test_spinoff_two_sided(preds.loc[preds["condition"] == condition, "age"].unstack().values.T,
                                              alternative="greater")
            x_locs = condition_data["week"].unqiue()
        for j, p_val in enumerate(p_vals):
            if p_val is None or np.isnan(p_val):
                continue
            y_min, y_max = ax.get_ylim()
            y_loc = (y_max - y_min) * 3 / 4 + y_min
            significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(x_locs[j], y_loc, significance, color=color, fontsize=18, ha="center")
        ax.set_ylabel("Lab test age difference from healthy pregnancy (years)", fontsize=14)
        finalize_weeks_postpartum_plots(ax)
        ax.axhline(y=0, color='k', linestyle='--', lw=1.)
        ax.legend(loc="upper right")
    return fig


def plot_model_weights(test_groups, model, color_mapping, per_top_row=5):
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    ordered_test_groups = sorted(test_groups.keys(), key=lambda x: len(test_groups[x]), reverse=True)
    max_num_tests = sum(len(test_groups[k]) for k in ordered_test_groups[:per_top_row])
    gs = GridSpec(5, max_num_tests, figure=fig)
    model_conf_int = model.conf_int().rename(columns={0: "bottom_ci", 1: "top_ci"})
    for i, group in enumerate(ordered_test_groups):
        tests = test_groups[group]
        if i < 5:
            k = 0
        else:
            k = 3
        # New line after 4 tests
        if i == 0 or i == 5:
            left = 0
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])
            ax.set_ylabel("clock weight (years)")
        else:
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        left = right
        test_order = model.params.loc[tests].sort_values().index
        display_name_tests = translate_long_to_short_lab(test_order)
        arbitrary_x_vals = range(len(tests))
        ax.bar(arbitrary_x_vals, model.params.loc[test_order], color=color_mapping.get(group, "gray"))
        whiskers = model_conf_int.loc[test_order].T - model.params.loc[test_order]
        whiskers.loc["bottom_ci"] *= -1
        ax.errorbar(arbitrary_x_vals, model.params.loc[test_order], yerr=whiskers, ls="", c="k", capsize=4)
        ax.set_xticks(arbitrary_x_vals)
        ax.set_xticklabels(display_name_tests, rotation=90, fontsize=10)
        ax.set_title(group, fontsize=14)
        # ax.set_ylim((-0.5, 3.5))
        remove_top_right_frame([ax])
        ax.grid(False)
    return fig


def plot_model_prediction(model, test_path=None, sample_size=1000, use_fill=False, normalize_pred=[-60,-40], exclude_points=None):
    test_names = model.params.index[model.params.index != "const"]
    if test_path is None:
        reference = sample_tests_mean_val(test_names, sample_size)
    else:
        reference = sample_tests_mean_val(test_names, sample_size, test_path)
    params = sample_regression_params(model.cov_params(), model.params, reference.shape[0])
    pred = pd.Series(data=np.einsum('ij,ij->i', add_constant(reference), params), index=reference.index).groupby(
        level=0).agg(age="mean", sd="std")
    if normalize_pred is not None:
        pred.loc[:, "age"] -= pred.loc[slice(*normalize_pred), "age"].mean()
    if exclude_points is not None:
        pred = pred[~pred.index.isin(exclude_points)]
    fig, ax = plt.subplots(1, 1)
    if use_fill:
        ax.plot(pred.index, pred["age"], '-')
        ax.fill_between(pred.index, pred["age"] - pred["sd"] * 1.96,
                        pred["age"] + pred["sd"] * 1.96, alpha=0.5)
    else:
        ax.errorbar(x=pred.index, y=pred["age"], yerr=pred["sd"] * 1.96, fmt='o-')
    ax.set_ylabel("Effective age difference (years)")
    finalize_weeks_postpartum_plots(ax)
    ax.legend(loc="lower right")
    return fig, ax, pred

