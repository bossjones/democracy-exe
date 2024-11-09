"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.stats._censored_data import CensoredData

if TYPE_CHECKING:
    ...
__all__ = ['ecdf', 'logrank']
@dataclass
class EmpiricalDistributionFunction:
    """An empirical distribution function produced by `scipy.stats.ecdf`

    Attributes
    ----------
    quantiles : ndarray
        The unique values of the sample from which the
        `EmpiricalDistributionFunction` was estimated.
    probabilities : ndarray
        The point estimates of the cumulative distribution function (CDF) or
        its complement, the survival function (SF), corresponding with
        `quantiles`.
    """
    quantiles: np.ndarray
    probabilities: np.ndarray
    _n: np.ndarray = ...
    _d: np.ndarray = ...
    _sf: np.ndarray = ...
    _kind: str = ...
    def __init__(self, q, p, n, d, kind) -> None:
        ...

    def evaluate(self, x):
        """Evaluate the empirical CDF/SF function at the input.

        Parameters
        ----------
        x : ndarray
            Argument to the CDF/SF

        Returns
        -------
        y : ndarray
            The CDF/SF evaluated at the input
        """
        ...

    def plot(self, ax=..., **matplotlib_kwargs): # -> list[Line2D]:
        """Plot the empirical distribution function

        Available only if ``matplotlib`` is installed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw the plot onto, otherwise uses the current Axes.

        **matplotlib_kwargs : dict, optional
            Keyword arguments passed directly to `matplotlib.axes.Axes.step`.
            Unless overridden, ``where='post'``.

        Returns
        -------
        lines : list of `matplotlib.lines.Line2D`
            Objects representing the plotted data
        """
        ...

    def confidence_interval(self, confidence_level=..., *, method=...): # -> ConfidenceInterval:
        """Compute a confidence interval around the CDF/SF point estimate

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval

        method : str, {"linear", "log-log"}
            Method used to compute the confidence interval. Options are
            "linear" for the conventional Greenwood confidence interval
            (default)  and "log-log" for the "exponential Greenwood",
            log-negative-log-transformed confidence interval.

        Returns
        -------
        ci : ``ConfidenceInterval``
            An object with attributes ``low`` and ``high``, instances of
            `~scipy.stats._result_classes.EmpiricalDistributionFunction` that
            represent the lower and upper bounds (respectively) of the
            confidence interval.

        Notes
        -----
        Confidence intervals are computed according to the Greenwood formula
        (``method='linear'``) or the more recent "exponential Greenwood"
        formula (``method='log-log'``) as described in [1]_. The conventional
        Greenwood formula can result in lower confidence limits less than 0
        and upper confidence limits greater than 1; these are clipped to the
        unit interval. NaNs may be produced by either method; these are
        features of the formulas.

        References
        ----------
        .. [1] Sawyer, Stanley. "The Greenwood and Exponential Greenwood
               Confidence Intervals in Survival Analysis."
               https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

        """
        ...



@dataclass
class ECDFResult:
    """ Result object returned by `scipy.stats.ecdf`

    Attributes
    ----------
    cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the empirical cumulative distribution function.
    sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the complement of the empirical cumulative
        distribution function.
    """
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction
    def __init__(self, q, cdf, sf, n, d) -> None:
        ...



def ecdf(sample: npt.ArrayLike | CensoredData) -> ECDFResult:
    """Empirical cumulative distribution function of a sample.

    The empirical cumulative distribution function (ECDF) is a step function
    estimate of the CDF of the distribution underlying a sample. This function
    returns objects representing both the empirical distribution function and
    its complement, the empirical survival function.

    Parameters
    ----------
    sample : 1D array_like or `scipy.stats.CensoredData`
        Besides array_like, instances of `scipy.stats.CensoredData` containing
        uncensored and right-censored observations are supported. Currently,
        other instances of `scipy.stats.CensoredData` will result in a
        ``NotImplementedError``.

    Returns
    -------
    res : `~scipy.stats._result_classes.ECDFResult`
        An object with the following attributes.

        cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            An object representing the empirical cumulative distribution
            function.
        sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            An object representing the empirical survival function.

        The `cdf` and `sf` attributes themselves have the following attributes.

        quantiles : ndarray
            The unique values in the sample that defines the empirical CDF/SF.
        probabilities : ndarray
            The point estimates of the probabilities corresponding with
            `quantiles`.

        And the following methods:

        evaluate(x) :
            Evaluate the CDF/SF at the argument.

        plot(ax) :
            Plot the CDF/SF on the provided axes.

        confidence_interval(confidence_level=0.95) :
            Compute the confidence interval around the CDF/SF at the values in
            `quantiles`.

    Notes
    -----
    When each observation of the sample is a precise measurement, the ECDF
    steps up by ``1/len(sample)`` at each of the observations [1]_.

    When observations are lower bounds, upper bounds, or both upper and lower
    bounds, the data is said to be "censored", and `sample` may be provided as
    an instance of `scipy.stats.CensoredData`.

    For right-censored data, the ECDF is given by the Kaplan-Meier estimator
    [2]_; other forms of censoring are not supported at this time.

    Confidence intervals are computed according to the Greenwood formula or the
    more recent "Exponential Greenwood" formula as described in [4]_.

    References
    ----------
    .. [1] Conover, William Jay. Practical nonparametric statistics. Vol. 350.
           John Wiley & Sons, 1999.

    .. [2] Kaplan, Edward L., and Paul Meier. "Nonparametric estimation from
           incomplete observations." Journal of the American statistical
           association 53.282 (1958): 457-481.

    .. [3] Goel, Manish Kumar, Pardeep Khanna, and Jugal Kishore.
           "Understanding survival analysis: Kaplan-Meier estimate."
           International journal of Ayurveda research 1.4 (2010): 274.

    .. [4] Sawyer, Stanley. "The Greenwood and Exponential Greenwood Confidence
           Intervals in Survival Analysis."
           https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

    Examples
    --------
    **Uncensored Data**

    As in the example from [1]_ page 79, five boys were selected at random from
    those in a single high school. Their one-mile run times were recorded as
    follows.

    >>> sample = [6.23, 5.58, 7.06, 6.42, 5.20]  # one-mile run times (minutes)

    The empirical distribution function, which approximates the distribution
    function of one-mile run times of the population from which the boys were
    sampled, is calculated as follows.

    >>> from scipy import stats
    >>> res = stats.ecdf(sample)
    >>> res.cdf.quantiles
    array([5.2 , 5.58, 6.23, 6.42, 7.06])
    >>> res.cdf.probabilities
    array([0.2, 0.4, 0.6, 0.8, 1. ])

    To plot the result as a step function:

    >>> import matplotlib.pyplot as plt
    >>> ax = plt.subplot()
    >>> res.cdf.plot(ax)
    >>> ax.set_xlabel('One-Mile Run Time (minutes)')
    >>> ax.set_ylabel('Empirical CDF')
    >>> plt.show()

    **Right-censored Data**

    As in the example from [1]_ page 91, the lives of ten car fanbelts were
    tested. Five tests concluded because the fanbelt being tested broke, but
    the remaining tests concluded for other reasons (e.g. the study ran out of
    funding, but the fanbelt was still functional). The mileage driven
    with the fanbelts were recorded as follows.

    >>> broken = [77, 47, 81, 56, 80]  # in thousands of miles driven
    >>> unbroken = [62, 60, 43, 71, 37]

    Precise survival times of the fanbelts that were still functional at the
    end of the tests are unknown, but they are known to exceed the values
    recorded in ``unbroken``. Therefore, these observations are said to be
    "right-censored", and the data is represented using
    `scipy.stats.CensoredData`.

    >>> sample = stats.CensoredData(uncensored=broken, right=unbroken)

    The empirical survival function is calculated as follows.

    >>> res = stats.ecdf(sample)
    >>> res.sf.quantiles
    array([37., 43., 47., 56., 60., 62., 71., 77., 80., 81.])
    >>> res.sf.probabilities
    array([1.   , 1.   , 0.875, 0.75 , 0.75 , 0.75 , 0.75 , 0.5  , 0.25 , 0.   ])

    To plot the result as a step function:

    >>> ax = plt.subplot()
    >>> res.cdf.plot(ax)
    >>> ax.set_xlabel('Fanbelt Survival Time (thousands of miles)')
    >>> ax.set_ylabel('Empirical SF')
    >>> plt.show()

    """
    ...

@dataclass
class LogRankResult:
    """Result object returned by `scipy.stats.logrank`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic (defined below). Its magnitude is the
        square root of the magnitude returned by most other logrank test
        implementations.
    pvalue : float ndarray
        The computed p-value of the test.
    """
    statistic: np.ndarray
    pvalue: np.ndarray
    ...


def logrank(x: npt.ArrayLike | CensoredData, y: npt.ArrayLike | CensoredData, alternative: Literal['two-sided', 'less', 'greater'] = ...) -> LogRankResult:
    r"""Compare the survival distributions of two samples via the logrank test.

    Parameters
    ----------
    x, y : array_like or CensoredData
        Samples to compare based on their empirical survival functions.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.

        The null hypothesis is that the survival distributions of the two
        groups, say *X* and *Y*, are identical.

        The following alternative hypotheses [4]_ are available (default is
        'two-sided'):

        * 'two-sided': the survival distributions of the two groups are not
          identical.
        * 'less': survival of group *X* is favored: the group *X* failure rate
          function is less than the group *Y* failure rate function at some
          times.
        * 'greater': survival of group *Y* is favored: the group *X* failure
          rate function is greater than the group *Y* failure rate function at
          some times.

    Returns
    -------
    res : `~scipy.stats._result_classes.LogRankResult`
        An object containing attributes:

        statistic : float ndarray
            The computed statistic (defined below). Its magnitude is the
            square root of the magnitude returned by most other logrank test
            implementations.
        pvalue : float ndarray
            The computed p-value of the test.

    See Also
    --------
    scipy.stats.ecdf

    Notes
    -----
    The logrank test [1]_ compares the observed number of events to
    the expected number of events under the null hypothesis that the two
    samples were drawn from the same distribution. The statistic is

    .. math::

        Z_i = \frac{\sum_{j=1}^J(O_{i,j}-E_{i,j})}{\sqrt{\sum_{j=1}^J V_{i,j}}}
        \rightarrow \mathcal{N}(0,1)

    where

    .. math::

        E_{i,j} = O_j \frac{N_{i,j}}{N_j},
        \qquad
        V_{i,j} = E_{i,j} \left(\frac{N_j-O_j}{N_j}\right)
        \left(\frac{N_j-N_{i,j}}{N_j-1}\right),

    :math:`i` denotes the group (i.e. it may assume values :math:`x` or
    :math:`y`, or it may be omitted to refer to the combined sample)
    :math:`j` denotes the time (at which an event occurred),
    :math:`N` is the number of subjects at risk just before an event occurred,
    and :math:`O` is the observed number of events at that time.

    The ``statistic`` :math:`Z_x` returned by `logrank` is the (signed) square
    root of the statistic returned by many other implementations. Under the
    null hypothesis, :math:`Z_x**2` is asymptotically distributed according to
    the chi-squared distribution with one degree of freedom. Consequently,
    :math:`Z_x` is asymptotically distributed according to the standard normal
    distribution. The advantage of using :math:`Z_x` is that the sign
    information (i.e. whether the observed number of events tends to be less
    than or greater than the number expected under the null hypothesis) is
    preserved, allowing `scipy.stats.logrank` to offer one-sided alternative
    hypotheses.

    References
    ----------
    .. [1] Mantel N. "Evaluation of survival data and two new rank order
           statistics arising in its consideration."
           Cancer Chemotherapy Reports, 50(3):163-170, PMID: 5910392, 1966
    .. [2] Bland, Altman, "The logrank test", BMJ, 328:1073,
           :doi:`10.1136/bmj.328.7447.1073`, 2004
    .. [3] "Logrank test", Wikipedia,
           https://en.wikipedia.org/wiki/Logrank_test
    .. [4] Brown, Mark. "On the choice of variance for the log rank test."
           Biometrika 71.1 (1984): 65-74.
    .. [5] Klein, John P., and Melvin L. Moeschberger. Survival analysis:
           techniques for censored and truncated data. Vol. 1230. New York:
           Springer, 2003.

    Examples
    --------
    Reference [2]_ compared the survival times of patients with two different
    types of recurrent malignant gliomas. The samples below record the time
    (number of weeks) for which each patient participated in the study. The
    `scipy.stats.CensoredData` class is used because the data is
    right-censored: the uncensored observations correspond with observed deaths
    whereas the censored observations correspond with the patient leaving the
    study for another reason.

    >>> from scipy import stats
    >>> x = stats.CensoredData(
    ...     uncensored=[6, 13, 21, 30, 37, 38, 49, 50,
    ...                 63, 79, 86, 98, 202, 219],
    ...     right=[31, 47, 80, 82, 82, 149]
    ... )
    >>> y = stats.CensoredData(
    ...     uncensored=[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24,
    ...                 25, 28,30, 33, 35, 37, 40, 40, 46, 48, 76, 81,
    ...                 82, 91, 112, 181],
    ...     right=[34, 40, 70]
    ... )

    We can calculate and visualize the empirical survival functions
    of both groups as follows.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> ax = plt.subplot()
    >>> ecdf_x = stats.ecdf(x)
    >>> ecdf_x.sf.plot(ax, label='Astrocytoma')
    >>> ecdf_y = stats.ecdf(y)
    >>> ecdf_y.sf.plot(ax, label='Glioblastoma')
    >>> ax.set_xlabel('Time to death (weeks)')
    >>> ax.set_ylabel('Empirical SF')
    >>> plt.legend()
    >>> plt.show()

    Visual inspection of the empirical survival functions suggests that the
    survival times tend to be different between the two groups. To formally
    assess whether the difference is significant at the 1% level, we use the
    logrank test.

    >>> res = stats.logrank(x=x, y=y)
    >>> res.statistic
    -2.73799...
    >>> res.pvalue
    0.00618...

    The p-value is less than 1%, so we can consider the data to be evidence
    against the null hypothesis in favor of the alternative that there is a
    difference between the two survival functions.

    """
    ...
