from typing import Iterable, Sequence, Union, Collection
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from matplotlib import colormaps as cms
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as lm

from .analyses import find_pregnancy_amplitude, find_labnorm_amplitude, z_test_spinoff_two_sided, find_diff_amplitude, \
    calculate_model_weights_ci, predict_age_in_pregnancy
from .clalit_parser import translate_long_to_short_lab, get_data_by_tests_and_field, get_condition_from_filename

_COLORS = ["red", "blue", "green", "orange", "brown"]


def remove_top_right_frame(axes: Iterable[Axes]):
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def remove_ax_labels_and_ticks(ax: Axes, remove_x: bool = False, remove_y: bool = False):
    ax.set_title('')
    if remove_x:
        ax.set_xlabel('')
        ax.set_xticks([])
    if remove_y:
        ax.set_yticks([])
        ax.set_ylabel('')


def finalize_weeks_postpartum_plots(ax: Axes, span_label: str = "Pregnancy"):
    """
    Add a gray span (rectangle) denoting the pregnancy period and rename the x-label "Week postpartum".
    Remove top right spines and ticks.
    """
    ax.set_xlabel("Week postpartum", fontsize=14)
    ax.axvspan(-38, 0, color='dimgray', alpha=0.1, zorder=-20, label=span_label)
    ax.grid(False)
    ax.set_xlim([-60, 80])
    remove_top_right_frame([ax])


# TODO reconsider name
def finalize_preg_panel(ax: Axes, span_label: str = "Pregnancy", xverts: tuple[float, ...] = ()):
    """
    Add a gray span (rectangle) denoting the pregnancy period and vertical dashed lines at the given xverts.
    :param ax:
    :param span_label:
    :param xverts:
    :return:
    """
    for xvert in xverts:
        ax.axvline(x=xvert, color='k', linestyle='--', lw=0.2, alpha=0.1)
    if xverts:
        ax.set_xticks(xverts)
    else:
        ax.set_xticks(np.arange(-50, 100, 50))
    finalize_weeks_postpartum_plots(ax, span_label)

    ax.tick_params(axis='both', which='major', labelsize=6)


# TODO delete
# def plot_series_seq(series_seq: Sequence[pd.Series], n_cols=6, ylabel=None) -> (plt.Figure, Axes):
#     """
#     Line plot a series index on the x-axis and values on the y-axis. Axes are not shared.
#     Index is expected to be numerical "weeks postpartum" (with deliver at 0) with a gray rectangle denoting pregnancy period.
#     The series MUST be named with a valid lab test file name.
#     It is possible to add an identical y label for all subplots.
#     :return: The figure and the axes object
#     """
#     axes, fig, n = _grid_by_seq(n_cols, series_seq)
#     for i, (ax, series) in enumerate(zip(axes.flat, series_seq)):
#         if not series.empty:
#             ax.plot(series.index, series.values, color="k", lw=0.5)
#             finalize_preg_panel(ax, "Pregnancy")
#             if ylabel is not None:
#                 ax.set_ylabel(ylabel, fontsize=10)
#         ax_name = translate_long_to_short_lab([series.name]).iloc[0]
#         ax.annotate(ax_name, (0.05, 1.05), xycoords='axes fraction', fontsize=10, color='k')
#         remove_x_labels = i < (n - n_cols)
#         # Remove y-label if series has no data or x label if it's not the last row
#         remove_ax_labels_and_ticks(ax, remove_x=remove_x_labels, remove_y=series.empty)
#     plt.tight_layout()
#     return fig, axes


def plot_df_seq(group_name: str, test_names: Sequence[str], df_seq: Sequence[Union[pd.Series, pd.DataFrame]],
                n_cols: int = 6, ylabel="age") -> plt.Figure:
    """
    Line plot a sequence of dataframes with the same index and give all the same y-label. The columns are the labels in the legend.
    Plot all the dataframe's columns in the same subplot.
    :param group_name: Name to appear on the figure
    :param test_names: Names of all the tests in the dataframe sequence. Length MUST be identical to df_seq. The test name annotates the subplot.
    :param df_seq: A sequence of dataframes with strictly numerical values.
    :param n_cols: Number of columns in the grid
    :param ylabel: A single y-label for all the subplots.
    :return: Tuple of the figure and the axes object
    """
    axes, fig, n = _create_grid_from_seq(n_cols, df_seq)
    for i, (ax, df) in enumerate(zip(axes.flat, df_seq)):
        remove_x_labels = i < (n - n_cols)
        plot_single_test_df(ax, df, test_names[i], ylabel, remove_x_labels)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"System: {group_name}", fontsize=16, y=1.0)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles=handles[:3], labels=labels[:3], loc="lower right")
    plt.tight_layout()
    return fig


def plot_single_test_df(ax: Axes, pd_obj: Sequence[Union[pd.DataFrame, pd.Series]], test_name: str, ylabel: str,
                        remove_x_labels: bool = False, annotation_coords: tuple[float, float] = (0.05, 1.05)):
    """
    Line plot either all the columns of a dataframe in a single subplot or a single series.
    :param ax: Axes object of the subplot
    :param pd_obj: Data to plot. Both index and values have to be numerical. If a series, does not have to be named.
    :param test_name: MUST be a valid lab test file name. The short version annotates the subplot.
    :param ylabel: Y-label text
    :param remove_x_labels: If the x labels and ticks should be removed from the x-axis.
    :param annotation_coords: Where to add the test name. Units are in 'axes fraction'. See matplotlib documentation Axes.annotate.
    :return:
    """
    if isinstance(pd_obj, pd.Series):
        ax.plot(pd_obj.index, pd_obj.values, color='r', lw=0.5)
    elif isinstance(pd_obj, pd.DataFrame):
        for j, col in enumerate(pd_obj.columns):
            ax.plot(pd_obj.index, pd_obj[col].values, color=_COLORS[j], lw=0.5, label=col)
    else:
        raise TypeError(f"pd_obj must be a pandas Series or DataFrame, got {type(pd_obj)} instead")
    finalize_preg_panel(ax, "Pregnancy", (-38, -25, 0, 10))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    ax_name = translate_long_to_short_lab([test_name]).iloc[0]
    ax.annotate(ax_name, annotation_coords, xycoords='axes fraction', fontsize=10, color='k', ha="right")
    remove_ax_labels_and_ticks(ax, remove_x=remove_x_labels)


# TODO consider delete
# def plot_linear(ax, line_x_points, rev_model, x_ticks=None, extra_points=None, xlabel="Age",
#                 ylabel="test units", c="k"):
#     if rev_model is not None:
#         line_y_points = rev_model(line_x_points)
#         ax.plot(line_x_points, line_y_points, color=c, lw=0.5)
#         if extra_points is not None:
#             ax.scatter(extra_points[0], extra_points[1], c=c, s=10)
#     if x_ticks is not None:
#         ax.set_xticks(x_ticks)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_facecolor('w')
#     remove_top_right_frame([ax])

# TODO consider delete
# def plot_tests_labnorm_age_on_ax_color_mapper(test_names: Sequence[str], ax, colormap=None, normalize=False,
#                                               color=None):
#     """
#
#     :param test_names:
#     :param ax:
#     :param colormap:
#     :param normalize:
#     :param color:
#     :return:
#     """
#     norm = Normalize(vmin=0, vmax=len(test_names) - 1, clip=True)
#     if colormap is None:
#         color_mapper = cm.ScalarMappable(norm=norm, cmap="turbo")
#     else:
#         color_mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
#     pd_series = []
#     for test_name in test_names:
#         labnorm_predict = predict_pregnancy_age_per_test(test_name, preconception_predictor=False)
#         # TODO extract, not in a plotting function
#         if normalize:
#             labnorm_predict = labnorm_predict / labnorm_predict.abs().max()
#         pd_series.append(labnorm_predict)
#     df = pd.concat(pd_series)
#     if color is None:
#         for i, test_name in enumerate(test_names):
#             ax.plot(labnorm_predict.index, labnorm_predict.values, c=color, lw=0.5, alpha=0.5)
#             coord = np.abs(labnorm_predict.values).argmax()
#             ax.text(labnorm_predict.index[coord], labnorm_predict.values[coord],
#                     translate_long_to_short_lab([test_name]).item(), ha='center', fontsize=10, color=color)
#     else:
#         plot_color_mapper_on_axes(ax, df, color_mapper=color_mapper)
#     finalize_preg_panel(ax, "Pregnancy")


def plot_color_mapper_on_axes(ax: Axes, df: pd.DataFrame, color_map: Union[str, Colormap] = "turbo"):
    test_names = df.columns
    test_names_annotate = translate_long_to_short_lab(test_names).values
    norm = Normalize(vmin=0, vmax=len(test_names) - 1, clip=True)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=color_map)
    for i, test_name in enumerate(test_names.values):
        ser = df[test_name]
        coord = np.abs(ser).argmax()
        c = color_mapper.to_rgba(np.array(i))
        ax.text(ser.index[coord], ser.values[coord], test_names_annotate[i], ha='center', fontsize=12, color=c)
        ax.plot(df.index, ser, c=c, lw=1, alpha=0.5)
        ax.set_ylabel("Effective age difference (year)")


# TODO delete
# def plot_tests_trends(json_path=os.path.join("csvs", "tests_trends.json"), normalize=True):
#     with open(json_path, "r") as f:
#         trends = json.load(f)
#     ncols = 2
#     nrows = len(trends)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.9, nrows * 3))
#     for i, (group_name, trend_tests) in enumerate(trends.items()):
#         for j, (trend, sig_tests) in enumerate(trend_tests.items()):
#             ax = axes[i][j]
#             for sig_test in sig_tests:
#                 if sig_test["is_sig"]:
#                     plot_tests_labnorm_age_on_ax_color_mapper(sig_test["tests"], ax, normalize=normalize)
#                 else:
#                     plot_tests_labnorm_age_on_ax_color_mapper(sig_test["tests"], ax, normalize=True,
#                                                               color="gray")
#         axes[i][0].annotate(group_name + ":", (0.05, 1.05), xycoords='axes fraction', fontsize=12, color='k',
#                             ha="right")
#
#     fig.suptitle("Trends", fontsize=16, y=1.0)
#     fig.subplots_adjust(wspace=0.1, hspace=1.8)
#     plt.tight_layout()
#     return fig, axes


def plot_quantile_diffs_histogram(test_names: Sequence[str], test_period: tuple[float, float] = (-40., 0.),
                                  labnorm_age_ref: tuple[int, int] = (20, 80), bins: int = 20) -> plt.Figure:
    """
    A histogram of difference in quantile change during pregnancy and aging.
    :param test_names: Sequence of all lab test names (valid file names)
    :param test_period: The period to consider for quantile diff for pregnant women. Default is pregnancy only from -40 weeks to 0 (delivery)
    :param labnorm_age_ref: A 2-tuple of ints. see `find_labnorm_amplitude` for more info.
    :param bins:
    :return:
    """
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
    return fig


def plot_quantile_diffs_pregnancy_ref(ax: Axes, test_names: Sequence[str],
                                      labnorm_age_ref: tuple[int, int] = (20, 80),
                                      test_period: tuple[float, float] = (-40., -1.),
                                      clalit_path: Union[None, str] = None):
    """
    Plot on a single subplot (represented by ax) the difference in quantile change during pregnancy (y-axis) and aging (x-axis).
    The difference in the y-axis is the maximum change in the weekly mean-quantile values during the test period.
    The difference in the x-axis is the difference between the median score at the top bracket of `labnorm_age_ref` and the bottom bracket. See `find_labnorm_amplitude` for more info.
    :param ax: The subplot to plot on
    :param test_names: Sequence of all lab test names (valid file names)
    :param labnorm_age_ref: A 2-tuple of ints. see `find_labnorm_amplitude` for more info.
    :param test_period: The pregnancy period to consider for the y-axis. Defaults to [-40., -1.]
    :param clalit_path: Path to the clalit data. If None, the default path is used.
    """
    for test_name in test_names:
        diff_labnorm, diff_labnorm_sd = find_labnorm_amplitude(labnorm_age_ref, test_name)
        pregnancy_diff_mean, pregnancy_diff_sd = find_pregnancy_amplitude(test_name, test_period,
                                                                          clalit_path=clalit_path)
        ax.errorbar(diff_labnorm, pregnancy_diff_mean, xerr=2 * diff_labnorm_sd, yerr=2 * pregnancy_diff_sd, c="k",
                    marker="o", markersize=5, capsize=2)
        ax.annotate(translate_long_to_short_lab([test_name]).item(), (diff_labnorm, pregnancy_diff_mean),
                    fontsize=13,
                    textcoords="offset points", xytext=(5, 5), ha='center')


def plot_diff_pathologies(ax: Axes, test_names: Sequence[str], pathology_path: str, healthy_path: str,
                          test_period: tuple[float, float] = (-40., 0.), labnorm_age_ref: tuple[int, int] = (20, 80),
                          c="k"):
    """
    Scatter plot with whiskers. On the y-axis the max difference (in quantiles) during pregnancy between some pathology and baseline "healthy".
    On the x-axis the difference in the median score in aging.
    :param ax: The subplot axes to plot on.
    :param test_names: Sequence of lab test names (valid file names) to plot.
    :param pathology_path: Path to Clalit directory with valid CSVs of the pathology data.
    :param healthy_path: Path to Clalit directory with valid CSVs of the healthy data.
    :param test_period: Time to consider for the pregnancy period in "weeks postpartum". Default is [-40., 0.] which is gestation.
    :param labnorm_age_ref: Old and young bracket to consider for the aging difference. Default is [20, 80].
    :param c: Color of the markers. See matplotlib's "Specifying colors" for more info.
    """
    for test_name in test_names:
        diff_labnrom, diff_labnorm_sd = find_labnorm_amplitude(labnorm_age_ref, test_name)
        path_diff_at_max_mean, path_diff_at_max_sd = find_diff_amplitude(test_name, pathology_path, test_period,
                                                                         clalit_path_base=healthy_path)
        ax.errorbar(diff_labnrom, path_diff_at_max_mean, xerr=diff_labnorm_sd * 2, yerr=2 * path_diff_at_max_sd,
                    marker="o", ms=5, c=c, capsize=2)
        ax.annotate(translate_long_to_short_lab([test_name]).item(), (diff_labnrom, path_diff_at_max_mean),
                    fontsize=13,
                    textcoords="offset points", xytext=(5, 5), ha='center')
    style_quartile_plots(ax, f"Reference quantile difference")


def plot_quantile_diff_pregnancy_model_weights(ax: Axes, test_names: Sequence[str],
                                               model: lm.RegressionResults,
                                               test_period: tuple[float, float] = (-40., 0.)):
    model_weights = calculate_model_weights_ci(model)
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
    style_quartile_plots(ax, f"Linear model weight", False)


def style_quartile_plots(ax: Axes, xlabel: str, same_scale: bool = True):
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


def grid_test_groups(test_groups_keys: Sequence[str], ncols: int) -> (plt.Figure, Sequence[Sequence[Axes]]):
    nrows = (len(test_groups_keys) - 1) // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    flattened_axs: Sequence[Axes] = axs.flatten()
    # Remove axes for any unused subplots
    for i in range(0, len(flattened_axs)):
        if i < len(test_groups_keys):
            flattened_axs[i].annotate(test_groups_keys[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=16)
        else:
            flattened_axs[i].axis('off')
    return fig, axs


def plot_diff_grid(test_groups: dict[str, Sequence[str]], labnorm_age_ref: tuple[int, int] = (20, 80),
                   test_period: tuple[float, float] = (-40, 0), ncols: int = 3, clalit_path: Union[str, None] = None,
                   margin: float = 0.05) -> plt.Figure:
    fig, axs = grid_test_groups(tuple(test_groups.keys()), ncols)
    flattened_axs: Sequence[Axes] = axs.flatten()
    for i, group in enumerate(test_groups):
        plot_quantile_diffs_pregnancy_ref(flattened_axs[i], test_groups[group], labnorm_age_ref, test_period,
                                          clalit_path)
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for ax in flattened_axs:
        min_x = min(min_x, ax.get_xlim()[0])
        max_x = max(max_x, ax.get_xlim()[1])
        min_y = min(min_y, ax.get_ylim()[0])
        max_y = max(max_y, ax.get_ylim()[1])
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax: Axes = axs[i, j]
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            style_quartile_plots(ax, f"Reference quantile difference")
            if j != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i != axs.shape[0] - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
    return fig


def plot_diff_grid_pathologies(test_groups: dict[str, Sequence[str]], pathology_path: str, healthy_path: str,
                               labnorm_age_ref: tuple[int, int] = (20, 80),
                               test_period: tuple[float, float] = (-40., 0.), ncols: int = 3, color="k"):
    fig, axs = grid_test_groups(tuple(test_groups.keys()), ncols)
    for i, group in enumerate(test_groups):
        row = i // axs.shape[0]
        col = i % axs.shape[1]
        ax: Axes = axs[row, col]
        plot_diff_pathologies(ax, test_groups[group], pathology_path, healthy_path, test_period,
                              labnorm_age_ref, color)
        if col != 0:
            ax.set_ylabel("")
        if row != axs.shape[0] - 1:
            ax.set_xlabel("")
    return fig


def plot_diff_grid_model(test_groups: dict[str, Sequence[str]], model: lm.RegressionResults,
                         test_period: tuple[float, float] = (-40., 0.), ncols: int = 3) -> plt.Figure:
    fig, axs = grid_test_groups(tuple(test_groups.keys()), ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        plot_quantile_diff_pregnancy_model_weights(flattened_axs[i], test_groups[group], model, test_period)
    return fig


def plot_groups_linear_prediction(test_groups: dict[str, Sequence[str]], model_parameters_ser: pd.Series,
                                  preconception_period: tuple[float, float] = (-60., -40.), ncols: int = 3,
                                  clalit_field: str = "val_mean", skip_range: Union[tuple[float, float], None] = None,
                                  colormap: Union[str, Colormap, None] = None):
    fig, axs = grid_test_groups(tuple(test_groups.keys()), ncols)
    flattened_axs = axs.flatten()
    for i, group in enumerate(test_groups):
        ax: Axes = flattened_axs[i]
        tests_to_consider = test_groups[group]
        group_df = get_data_by_tests_and_field(tests_to_consider, clalit_field)
        considered_parameters = model_parameters_ser[tests_to_consider]
        prediction = group_df @ considered_parameters
        prediction -= prediction.loc[preconception_period[0]:preconception_period[1]].mean()
        individual_predictions = group_df * considered_parameters
        individual_predictions -= individual_predictions.loc[preconception_period[0]:preconception_period[1]].mean()
        if skip_range is not None:
            prediction = prediction.loc[~prediction.index.isin(prediction.loc[slice(*skip_range)].index)]
            individual_predictions = individual_predictions.loc[
                ~individual_predictions.index.isin(individual_predictions.loc[slice(*skip_range)].index)]
        if colormap is not None:
            plot_color_mapper_on_axes(ax, individual_predictions, colormap)
        else:
            plot_color_mapper_on_axes(ax, individual_predictions)
        ax.plot(prediction.index, prediction.values, color='k', lw=1,
                label="Group contribution" if i == 0 else None)
        finalize_weeks_postpartum_plots(ax)
    fig.legend(loc="lower right", fontsize=12)
    return fig, axs


# TODO delete
# def plot_test_with_linear_models(group_name, test_names: Sequence[str]):
#     ncols = 3
#     nrows = len(test_names)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2))
#     for i, test_name in enumerate(test_names):
#         labnorm_model, (labnorm_val_train, labnorm_age_train) = model_labnorm_linreg(test_name)
#         age_predictions_df = predict_pregnancy_age_per_test(test_name, labnorm_predictor=labnorm_model,
#                                                             preconception_predictor=False)
#         median_test_values = get_clalit_data(test_name)["val_50"]
#         edge_ages = [15, 61]
#         age_ticks = np.arange(edge_ages[0], edge_ages[1], 15)
#         reverse_labnorm_model = reverse_1d_linear_model(labnorm_model)
#         labnorm_age_to_median = find_median_for_lab(test_name)
#         labnrom_r_pvalue = pearsonr(labnorm_age_to_median.index, labnorm_age_to_median.values).pvalue
#         labnorm_spearman_stat = spearmanr(labnorm_age_to_median.index, labnorm_age_to_median.values).statistic
#         for j in range(ncols):
#             ax: Axes = axes[i][j]
#             if j == 0:
#                 plot_single_test_df(ax, age_predictions_df, test_name, "age")
#                 if i == 0:
#                     ax.set_title("Predicted Age", fontsize=12, y=1.08)
#             elif j == 1:
#                 ax.plot(median_test_values.index, median_test_values.values, color="k", lw=0.5)
#                 finalize_preg_panel(ax, "Pregnancy")
#                 if i == 0:
#                     ax.set_title("Test Value", fontsize=12, y=1.08)
#             elif j == 2:
#                 # Colors indexing is bad and likely cause confusion with each change
#                 plot_linear(ax, edge_ages, reverse_labnorm_model, x_ticks=age_ticks,
#                             extra_points=(labnorm_age_train, labnorm_val_train), c=_COLORS[0])
#                 ax.plot(labnorm_age_to_median.index, labnorm_age_to_median.values, c=_COLORS[0], lw=0.5, alpha=0.3)
#                 ax.annotate(f"Pear. p-val: {labnrom_r_pvalue:.2f}\nSpearman: {labnorm_spearman_stat:.2f}", (0.05, 0.95),
#                             va="bottom",
#                             xycoords='axes fraction', fontsize=8, color='k')
#                 if i == 0:
#                     ax.set_title("Labnorm Linear Model", fontsize=12, y=1.08)
#     fig.suptitle(f"System: {group_name}", fontsize=16, y=1.0)
#     fig.subplots_adjust(wspace=0.1, hspace=0.1)
#     plt.tight_layout()
#     return fig, axes

def plot_age_acceleration_by_lin_reg(model: lm.RegressionResults, relative_path: str, compared_paths: Collection[str],
                                     sample_size: int = 1000, exclude_points: Union[Iterable[float], None] = None,
                                     compared_color_map: Union[dict[str, str], None] = None,
                                     p_val_bins: Union[Sequence[float], None] = None, use_fill: bool = False,
                                     y_limit: Union[tuple[float, float], None] = None,
                                     y_ticks: Union[Iterable[float], None] = None, fix_outliers: bool = False,
                                     fig_width: float = 12, subplot_height: float = 5.8) -> plt.Figure:
    if compared_color_map is not None and len(compared_color_map) != len(compared_paths):
        raise ValueError("compared_color_map must have the same length as compared_paths")
    # The compared medical conditions are either given in the key map or parsed from the paths
    conditions = list(compared_color_map.keys()) if compared_color_map is not None else [None] * len(compared_paths)
    # agg_preds = {}
    unagg_preds = {}
    for i, compared_path in enumerate(compared_paths):
        agg_pred, unagg_pred = predict_age_in_pregnancy(model, relative_path, sample_size, exclude_points, fix_outliers,
                                                        compared_path)
        if conditions[i] is None:
            conditions[i] = get_condition_from_filename(compared_path)
        condition = conditions[i]
        # agg_preds[condition] = agg_pred
        unagg_preds[condition] = unagg_pred
    unagg_preds = pd.concat(unagg_preds, names=['condition'])
    # Figure preparation
    if compared_color_map is None:
        compared_color_map = {conditions[i]: color for i, color in
                              enumerate(cms.tab10(np.linspace(0, 1, len(conditions))))}
    if use_fill:
        err_style = "band"
        err_kws = {"alpha": 0.2}
    else:
        err_style = "bars"
        err_kws = {"fmt": "o-"}
    g = sns.FacetGrid(data=unagg_preds.reset_index(), row="condition", hue="condition",
                      height=subplot_height, aspect=fig_width / subplot_height, palette=compared_color_map)
    g.map(sns.lineplot, "week", "age", err_style=err_style, err_kws=err_kws, errorbar=("sd", 1.96))
    # Add line and edit all y-axis parameters
    g.refline(y=0, color='k', linestyle='--', lw=1.)
    if y_limit is not None:
        g.set(ylim=y_limit, yticks=np.arange(y_limit[0], y_limit[1], 5) if y_ticks is None else y_ticks)
    g.set_ylabels("Effective age difference (years)", fontsize=14)
    # Denote significantly greater than zero with asterisks. Compute bin limits and p-vals.
    for condition, unagg_cond in unagg_preds.groupby(level="condition"):
        ax = g.axes_dict[condition]  # Prone to failure: axes_dict is not guaranteed to have the keys as the row names
        if p_val_bins is not None:
            x_locs = []  # Where to write the text denoting significance
            p_vals = []
            first_measurement_in_bin = None
            for j in range(0, len(p_val_bins) - 1):
                if first_measurement_in_bin is None:
                    first_measurement_in_bin = p_val_bins[j]  # Bottom bound of the bin
                right = p_val_bins[j + 1]  # Top bound of the bin
                week_level_indices = unagg_cond.index.get_level_values("week")
                bin_vals = unagg_cond.loc[
                    (week_level_indices >= first_measurement_in_bin) & (week_level_indices <= right)]
                # Data points are not necessarily aligned with the bin limits, pick the next index
                first_measurement_in_bin = week_level_indices[
                    week_level_indices > bin_vals.index.get_level_values("week").max()].min()
                p_vals.append(z_test_spinoff_two_sided(bin_vals.values, alternative="greater"))
                x_locs.append((p_val_bins[j] + right) / 2)  # Mid-bin
                if j != 0:  # don't add vertical line for the xmin value
                    # Separate the bins visually
                    ax.axvline(x=p_val_bins[j], color="gray", linestyle="--", lw=2, alpha=0.3)
            p_vals = false_discovery_control(p_vals)  # FDR between the bins
        else:  # Do not bin, use all points
            p_vals = z_test_spinoff_two_sided(unagg_cond.unstack().values.T, alternative="greater")
            x_locs = unagg_preds.get_level_values("week").unqiue()
        for j, p_val in enumerate(p_vals):
            if p_val is None or np.isnan(p_val):
                continue
            y_min, y_max = ax.get_ylim()
            y_loc = (y_max - y_min) * 3 / 4 + y_min
            significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(x_locs[j], y_loc, significance, color=compared_color_map[condition], fontsize=18, ha="center")
        finalize_weeks_postpartum_plots(ax)
        ax.legend(loc="upper right")  # No choice but to use a legend per subplot
        # Larger text on the x tick labels
        if ax.get_xticklabels():
            ax.set_xticks(ax.get_xticks())  # Supress warning
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    # g.add_legend(loc='lower center', ncols=len(conditions), fontsize=14)  # For some reason, this comes out ugly
    return g.figure


def plot_model_weights(test_groups: dict[str, Sequence[str]], model: lm.RegressionResults,
                       color_mapping: dict[str, str], per_top_row: int = 5,
                       y_limit: Union[None, tuple[float, float]] = None) -> plt.Figure:
    """
    Bar plot of the model weights. Each group is a subplot with the tests as the x-axis and the weights as the y-axis.
    :param test_groups: Mapping from group name to annotate as text to the lab tests, which are valid lab test file names.
    :param model: The linear model to plot the weights of.
    :param color_mapping: Mapping from group name to color to use for the bars, see matplotlib's "Specifying colors" for more info about colors.
    :param per_top_row: Number of groups to plot in the top row. The rest will be in the bottom row.
    :param y_limit: A tuple with first element as lower limit and second element as upper limit of the y-axis of all subplots.
    :return: The figure object of the plot
    """
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    # Order groups by number of lab tests per group, descending
    ordered_test_groups = sorted(test_groups.keys(), key=lambda x: len(test_groups[x]), reverse=True)
    max_num_tests = sum(
        len(test_groups[k]) for k in ordered_test_groups[:per_top_row])  # Number of tests in the top row
    # Each bar is occupying a slot in the grid
    gs = GridSpec(5, max_num_tests, figure=fig)
    model_conf_int = calculate_model_weights_ci(model)
    # Plot all tests divided by groups
    left = -1  # init
    for i, group in enumerate(ordered_test_groups):
        tests = test_groups[group]
        if i < per_top_row:
            k = 0
        else:
            k = 3  # A nice space between the rows
        # New line after `per_top_row` tests
        if i == 0 or i == per_top_row:
            left = 0
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])  # Allot 3 rows
            ax.set_ylabel("LabAge clock weight (years)")
        else:  # Annoying repeat, but I couldn't put the code outside the condition - setting the yticks and removing the left spine must happen after creating the axes object
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        left = right
        test_order = model.params.loc[tests].sort_values().index  # Plot smallest to largest
        display_name_tests = translate_long_to_short_lab(test_order)
        arbitrary_x_vals = range(len(tests))  # Location on the x-axis, values don't matter because it's a bar plot.
        ax.bar(arbitrary_x_vals, model.params.loc[test_order], color=color_mapping.get(group, "gray"))
        whiskers = model_conf_int.loc[test_order, ["bottom_ci", "upper_ci"]] - model.params.loc[test_order, "value"]
        ax.errorbar(arbitrary_x_vals, model_conf_int.loc[test_order, "value"], yerr=whiskers.T, ls="", c="k", capsize=4)
        ax.set_xticks(arbitrary_x_vals)
        ax.set_xticklabels(display_name_tests, rotation=90, fontsize=14)
        ax.set_title(group, fontsize=15)
        remove_top_right_frame([ax])
        ax.grid(False)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)  # Yes, weird but necessary for fontsize
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    return fig


def plot_model_prediction(model: lm.RegressionResults, test_path: Union[None, str] = None, sample_size: int = 1000,
                          exclude_points: Union[None, Iterable[float]] = None, fix_outliers: bool = False,
                          use_fill=False, baseline: Union[None, tuple[float, float]] = (-60., -40.), ax=None,
                          color=None, label=None, alpha=0.5) -> (Union[None, plt.Figure], Axes):
    """
    Line plot with error bars an age prediction on the pregnancy data.
    :param model: The linear model to predict the age acceleration.
    :param test_path: Path to the test data, a directory with valid lab test CSV files. If None, the default path is used.
    :param sample_size: Number of samples to draw from the model. Greater size yields more accurate error bars but takes more time.
    :param exclude_points: Weeks to exclude from the prediction. If None, no points are excluded.
    :param fix_outliers: If True, outliers are fixed by estimating the mean and standard deviation for the outliers.
    :param use_fill: Instead of error bars, fill the area between the upper and lower bounds of the prediction.
    :param baseline: Subtract the mean of the prediction in this range. If None, prediction is given as is.
    :param ax: An axes to draw the plot on. If None, a new figure is created (and returned).
    :param color: Line and error bar/fill color.
    :param label: Name of the line in the legend.
    :param alpha: Transparency (1 no transparency, 0 full transparency) of the fill color. If `use_fill=False` the value is unused.
    :return:
    """
    pred, unagg_pred = predict_age_in_pregnancy(model, test_path, sample_size, exclude_points, fix_outliers)
    if baseline is not None:
        unagg_pred.loc[:] -= unagg_pred.loc[slice(*baseline)].mean()
    # TODO delete after testing
    # if ax is None:
    #     fig, ax = plt.subplots()
    # else:
    #     fig = None
    if use_fill:
        err_style = "band"
        err_kws = {"alpha": 2 * alpha}
    #     color = ax.plot(pred.index, pred["age"], '-', color=color, label=label, alpha=2 * alpha)[0].get_color()
    #     ax.fill_between(pred.index, pred["age"] - pred["sd"] * 1.96,
    #                     pred["age"] + pred["sd"] * 1.96, alpha=alpha, color=color)
    else:
        err_style = "bars"
        err_kws = {"fmt": "o-"}
    #     ax.errorbar(x=pred.index, y=pred["age"], yerr=pred["sd"] * 1.96, fmt='o-')
    ret_ax = sns.lineplot(unagg_pred.reset_index(), x="week", y="age", ax=ax, color=color, label=label,
                          errorbar=("sd", 1.96), err_style=err_style, alpha=alpha, err_kws=err_kws)
    if ax is None:  # New plot
        ret_ax.axhline(y=0, color='k', linestyle='-', lw=2., alpha=0.5)
        ret_ax.set_ylabel("Mothers mean age acceleration (years)")
        finalize_weeks_postpartum_plots(ret_ax)
    ret_ax.legend(loc="lower right")
    return plt.gcf(), ret_ax


def _create_grid_from_seq(ncols, series_seq, panel_size=2.) -> (Axes, plt.Figure, int):
    n = len(series_seq)
    nrows = (n - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size, nrows * panel_size))
    # Remove axes for any unused subplots
    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis('off')
    return axes, fig, n
