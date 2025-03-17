import numpy as np
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d


class LinearInterpDistribution(rv_continuous):
    """
    A custom distribution made by linearly interpolating between quantiles.
    """
    def __init__(self, values: np.ndarray, quantiles: np.ndarray, *args, **kwargs):
        """
        Values and quantiles MUST be ordered in ascending order and have the same size.
        If the quantiles (0,1) are not specified, extrapolate using nearset values.
        Values outside the range are clipped.
        :param values: The real values of the distribution
        :param quantiles: The quantile values, a discrete representation of the CDF.
        :param args: See rv_continuous
        :param kwargs: See rv_continuous
        """
        super().__init__(*args, **kwargs)
        self.values = values
        self.quantiles = quantiles
        # Check - really need to normalize mean and variance?
        self.__ppf_interp = interp1d(self.quantiles, self.values, bounds_error=False, fill_value=(self.values.min(), self.values.max()))
        self.__cdf_interp = interp1d(self.values, self.quantiles, bounds_error=False, fill_value=(0., 1.))

    def _ppf(self, q, *args):
        return self.__ppf_interp(np.clip(q, 0., 1.))

    def _cdf(self, x, *args):
        return np.clip(self.__cdf_interp(x), 0., 1.)


