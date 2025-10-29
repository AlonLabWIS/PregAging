#new Glen Oct 2025
#for doing marginal regression, saving results to use Ron's functions

import pandas as pd
from typing import Iterable, Sequence, Union, Collection
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from matplotlib import colormaps as cms
from matplotlib.gridspec import GridSpec
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt

from .clalit_parser import translate_long_to_short_lab
from .plotting import remove_top_right_frame


def FitMarginalOLS(y,X,
			addConst=True):
	"""
	Fits marginal OLS models for each supplied model in tuple, returns dictionary of essential values
	:param y: variable to predict
	:param X: all variables to predict with (dataframe)
	:param addConst: include intercept in regression?
	"""
    
	vars_ = list(X.columns)
	params     = pd.Series(index=vars_,dtype="float")
	pvalues    = pd.Series(index=vars_,dtype="float")
	conf_int   = pd.DataFrame(columns=["bottom_ci", "upper_ci","value"],index=vars_,dtype="float")
	cov_params = pd.DataFrame(0.0,columns=vars_,index=vars_)
	for i,v in enumerate(vars_):
		preds = X[[v]].copy()
		if addConst:    preds = add_constant(preds, has_constant="add")
		mod = OLS(y, preds,missing="drop").fit()
		params.loc[v] = mod.params[v]
		pvalues.loc[v] = mod.pvalues[v]
		mod_ci = mod.conf_int().loc[v]
		conf_int.loc[v,"bottom_ci"] = mod_ci[0]
		conf_int.loc[v,"upper_ci"] = mod_ci[1]
		conf_int.loc[v,"value"] = mod.params[v]
		cov_params.loc[v,v] = mod.cov_params().loc[v,v]
 
	return  {'params': params, 'pvalues': pvalues, 'conf_int':conf_int, 'cov_params':cov_params}


def plot_model_weights_marginal(test_groups: dict[str, Sequence[str]], 
                        model_conf_int,model_params,
                       color_mapping: dict[str, str], per_top_row: int = 5, num_row: int=2,
                       y_limit: Union[None, tuple[float, float]] = None) -> plt.Figure:
    """
    Bar plot of the model weights. Each group is a subplot with the tests as the x-axis and the weights as the y-axis.
    :param test_groups: Mapping from group name to annotate as text to the lab tests, which are valid lab test file names.
    :param model_conf_int: pandas.core.frame.DataFrame, number of params x 3 columns: bottom_ci, upper_ci and value
    :param model_params: 
pandas.core.series.Series of length = number params
    :param color_mapping: Mapping from group name to color to use for the bars, see matplotlib's "Specifying colors" for more info about colors.
    :param per_top_row: Number of groups to plot in the top row. The rest will be in the bottom row.
    :param y_limit: A tuple with first element as lower limit and second element as upper limit of the y-axis of all subplots.
    :return: The figure object of the plot
    """
    fig = plt.figure(layout="constrained", figsize=(20, 10*num_row/2))
    # Order groups by number of lab tests per group, descending
    ordered_test_groups = sorted(test_groups.keys(), key=lambda x: len(test_groups[x]), reverse=True)
    max_num_tests = sum(
        len(test_groups[k]) for k in ordered_test_groups[:per_top_row])  # Number of tests in the top row
    # Each bar is occupying a slot in the grid
    gs = GridSpec(3*num_row-1, max_num_tests, figure=fig)
    # Plot all tests divided by groups
    left = -1  # init
    for i, group in enumerate(ordered_test_groups):
        tests = test_groups[group]
        k = 3*( i // per_top_row) # A nice space between the rows
        # New line after `per_top_row` tests
        if i % per_top_row == 0:
            left = 0
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])  # Allot 3 rows
            ax.set_ylabel("Age association strength (years/sd)") 
        else:  # Annoying repeat, but I couldn't put the code outside the condition - setting the yticks and removing the left spine must happen after creating the axes object
            right = left + len(tests)
            ax = fig.add_subplot(gs[k:k + 2, left:right])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        left = right
        test_order = model_params.loc[tests].sort_values().index  # Plot smallest to largest
        display_name_tests = translate_long_to_short_lab(test_order)
        arbitrary_x_vals = range(len(tests))  # Location on the x-axis, values don't matter because it's a bar plot.
        ax.bar(arbitrary_x_vals, model_params.loc[test_order], color=color_mapping.get(group, "gray"))
        whiskers = model_conf_int.loc[test_order, ["bottom_ci", "upper_ci"]].T - model_params[test_order]
        whiskers.loc["bottom_ci"] *= -1  # Bottom ci is negative, and matplotlib expects positive values (taking the error bar down by the same amount)
        ax.errorbar(arbitrary_x_vals, model_conf_int.loc[test_order, "value"], yerr=whiskers, ls="", c="k", capsize=4)
        ax.set_xticks(arbitrary_x_vals)
        ax.set_xticklabels(display_name_tests, rotation=90, fontsize=14)
        ax.set_title(group, fontsize=15)
        remove_top_right_frame([ax])
        ax.grid(False)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.set_yticks(ax.get_yticks())  # Supress warning about setting yticklabels w/o setting yticks
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)  # Yes, weird but necessary for fontsize
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
    return fig

