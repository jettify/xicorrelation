from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, rankdata

from ._version import version as _version

__all__ = ("xicorr", "XiCorrResult")
__version__ = _version


class XiCorr(NamedTuple):
    xi: float
    fr: npt.NDArray[np.float64]
    cu: float


class XiCorrResult(NamedTuple):
    correlation: float
    pvalue: Optional[float]


def _xicorr(x: npt.NDArray, y: npt.NDArray) -> XiCorr:
    # Ported from original R implementation.
    # https://github.com/cran/XICOR/blob/master/R/calculateXI.R
    n = x.size
    PI = rankdata(x, method="average")
    fr = rankdata(y, method="average") / n
    gr = rankdata(-y, method="average") / n
    cu = np.mean(gr * (1 - gr))
    A1 = np.abs(np.diff(fr[np.argsort(PI, kind="quicksort")])).sum() / (2 * n)
    xi = 1.0 - A1 / cu
    return XiCorr(xi, fr, cu)


def xicorr(x: npt.ArrayLike, y: npt.ArrayLike, ties=True) -> XiCorrResult:
    """
    Compute the cross rank increment correlation coefficient xi.
    This function computes the xi coefficient between two vectors x and y.


    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    ties : bool, optional
        If ties is True, the algorithm assumes that the data has ties and
        employs the more elaborated theory for calculating s.d. and P-value.
        Otherwise, it uses the simpler theory. There is no harm in putting
        ties = True even if there are no ties.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       P-values computed by the asymptotic theory.

    See Also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.


    Example:
        >>> from xicorrelation import xicorr
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [1, 4, 9, 16, 25]
        >>> xi, pvalue = xicorr(x, y)
        >>> print(xi, pvalue)

    References
    ----------
    .. [1] Chatterjee, S., "A new coefficient of correlation",
           https://arxiv.org/abs/1909.10140, 2020.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> xi, p_value, _ = xicorr(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748
    """
    # https://git.io/JSIlN
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `xicorr` must be of the same "
            f"size, found x-size {x.size} and y-size {y.size}"
        )
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        return XiCorrResult(np.nan, np.nan)

    r = _xicorr(x, y)
    xi = r.xi
    fr = r.fr
    CU = r.cu

    pvalue = None
    # https://git.io/JSIlM
    n = x.size
    if not ties:
        # sd = np.sqrt(2.0 / (5.0 * n))
        pvalue = 1.0 - norm.cdf(np.sqrt(n) * xi / np.sqrt(2.0 / 5.0))
    else:
        qfr = np.sort(fr)
        ind = np.arange(1, n + 1)
        ind2 = 2 * n - 2 * ind + 1

        ai = np.mean(ind2 * qfr * qfr) / n
        ci = np.mean(ind2 * qfr) / n
        cq = np.cumsum(qfr)
        m = (cq + (n - ind) * qfr) / n
        b = np.mean(m ** 2)
        v = (ai - 2.0 * b + ci ** 2) / (CU ** 2)

        # sd = np.sqrt(v / n)
        pvalue = 1.0 - norm.cdf(np.sqrt(n) * xi / np.sqrt(v))
    return XiCorrResult(xi, pvalue)
