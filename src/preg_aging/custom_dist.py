import numpy as np
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d


class LinearInterpDistribution(rv_continuous):
    def __init__(self, values, quantiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = values.values
        self.quantiles = quantiles.values
        # Check - really need to normalize mean and variance?
        self.__ppf_interp = interp1d(self.quantiles, self.values, bounds_error=False, fill_value=(self.values.min(), self.values.max()))
        self.__cdf_interp = interp1d(self.values, self.quantiles, bounds_error=False, fill_value=(0., 1.))

    def _ppf(self, q, *args):
        return self.__ppf_interp(np.clip(q, 0., 1.))

    def _cdf(self, x, *args):
        return np.clip(self.__cdf_interp(x), 0., 1.)


