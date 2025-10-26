#new Glen Oct 2025
#for doing marginal regression, saving results to use Ron's functions

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

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
