"""
This type stub file was generated by pyright.
"""

class OddsRatioResult:
    """
    Result of `scipy.stats.contingency.odds_ratio`.  See the
    docstring for `odds_ratio` for more details.

    Attributes
    ----------
    statistic : float
        The computed odds ratio.

        * If `kind` is ``'sample'``, this is sample (or unconditional)
          estimate, given by
          ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
        * If `kind` is ``'conditional'``, this is the conditional
          maximum likelihood estimate for the odds ratio. It is
          the noncentrality parameter of Fisher's noncentral
          hypergeometric distribution with the same hypergeometric
          parameters as `table` and whose mean is ``table[0, 0]``.

    Methods
    -------
    confidence_interval :
        Confidence interval for the odds ratio.
    """
    def __init__(self, _table, _kind, statistic) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def confidence_interval(self, confidence_level=..., alternative=...): # -> ConfidenceInterval:
        """
        Confidence interval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.
            Default is 0.95 (meaning 95%).

        alternative : {'two-sided', 'less', 'greater'}, optional
            The alternative hypothesis of the hypothesis test to which the
            confidence interval corresponds. That is, suppose the null
            hypothesis is that the true odds ratio equals ``OR`` and the
            confidence interval is ``(low, high)``. Then the following options
            for `alternative` are available (default is 'two-sided'):

            * 'two-sided': the true odds ratio is not equal to ``OR``. There
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``high < OR`` or ``low > OR``.
            * 'less': the true odds ratio is less than ``OR``. The ``low`` end
              of the confidence interval is 0, and there is evidence against
              the null hypothesis at  the chosen `confidence_level` if
              ``high < OR``.
            * 'greater': the true odds ratio is greater than ``OR``.  The
              ``high`` end of the confidence interval is ``np.inf``, and there
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``low > OR``.

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When `kind` is ``'conditional'``, the limits of the confidence
        interval are the conditional "exact confidence limits" as described
        by Fisher [1]_. The conditional odds ratio and confidence interval are
        also discussed in Section 4.1.2 of the text by Sahai and Khurshid [2]_.

        When `kind` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [2]_, section 3.1.3.2,
        or [3]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [3] Alan Agresti, An Introduction to Categorical Data Analysis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
        ...
    


def odds_ratio(table, *, kind=...): # -> OddsRatioResult:
    r"""
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.

    Returns
    -------
    result : `~scipy.stats._result_classes.OddsRatioResult` instance
        The returned object has two computed attributes:

        statistic : float
            * If `kind` is ``'sample'``, this is sample (or unconditional)
              estimate, given by
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.

        The object has the method `confidence_interval` that computes
        the confidence interval of the odds ratio.

    See Also
    --------
    scipy.stats.fisher_exact
    relative_risk

    Notes
    -----
    The conditional odds ratio was discussed by Fisher (see "Example 1"
    of [1]_).  Texts that cover the odds ratio include [2]_ and [3]_.

    .. versionadded:: 1.10.0

    References
    ----------
    .. [1] R. A. Fisher (1935), The logic of inductive inference,
           Journal of the Royal Statistical Society, Vol. 98, No. 1,
           pp. 39-82.
    .. [2] Breslow NE, Day NE (1980). Statistical methods in cancer research.
           Volume I - The analysis of case-control studies. IARC Sci Publ.
           (32):5-338. PMID: 7216345. (See section 4.2.)
    .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classified as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of a case occurring in the exposed group,
    and ``b/d`` as the odds of a case occurring in the unexposed group.
    The sample odds ratio is the ratio of these odds.  If the odds ratio
    is greater than 1, it suggests that there is a positive association
    between being exposed and being a case.

    Interchanging the rows or columns of the contingency table inverts
    the odds ratio, so it is important to understand the meaning of labels
    given to the rows and columns of the table when interpreting the
    odds ratio.

    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    The question we ask is "Is there evidence that the aspirin reduces the
    risk of ischemic stroke?"

    Compute the odds ratio:

    >>> from scipy.stats.contingency import odds_ratio
    >>> res = odds_ratio([[176, 230], [21035, 21018]])
    >>> res.statistic
    0.7646037659999126

    For this sample, the odds of getting an ischemic stroke for those who have
    been taking aspirin are 0.76 times that of those
    who have received the placebo.

    To make statistical inferences about the population under study,
    we can compute the 95% confidence interval for the odds ratio:

    >>> res.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=0.6241234078749812, high=0.9354102892100372)

    The 95% confidence interval for the conditional odds ratio is
    approximately (0.62, 0.94).

    The fact that the entire 95% confidence interval falls below 1 supports
    the authors' conclusion that the aspirin was associated with a
    statistically significant reduction in ischemic stroke.
    """
    ...

