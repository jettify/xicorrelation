import random

import numpy as np
import pytest

from xicorrelation import xicorr


@pytest.fixture
def quartet():
    x_1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    x_2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    x_3 = [0, 0, 0, 0, 1, 1, 2, 3]
    x_4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y_1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y_2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74]
    y_3 = [0, 1, 2, 3, 4, 5, 6, 7]
    y_4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89]
    anscombes_quartet = {
        "x_1": x_1,
        "x_2": x_2,
        "x_3": x_3,
        "x_4": x_4,
        "y_1": y_1,
        "y_2": y_2,
        "y_3": y_3,
        "y_4": y_4,
    }

    return anscombes_quartet


def test_xi_correlations(quartet):
    q = quartet

    assert xicorr(q["x_1"], q["y_1"]).correlation == pytest.approx(
        0.275, rel=10 ** -6
    )
    assert xicorr(q["x_2"], q["y_2"]).correlation == pytest.approx(
        0.6, rel=10 ** -6
    )
    assert xicorr(q["x_3"], q["y_3"]).correlation == pytest.approx(
        0.6666666666666667, rel=10 ** -6
    )
    assert xicorr(q["x_4"], q["y_4"]).correlation == pytest.approx(
        0.175, rel=10 ** -6
    )


def test_xi_correlations_no_ties(quartet):
    q = quartet
    x = xicorr(q["x_1"], q["y_1"], ties=False)
    assert x.correlation == pytest.approx(0.275, rel=10 ** -6)


def test_p_val_asymptotic(quartet):
    random.seed(2020)
    np.random.seed(2020)
    q = quartet
    # values taken from R code
    assert xicorr(q["x_1"], q["y_1"], ties=True).pvalue == pytest.approx(
        0.0784155644
    )
    assert xicorr(q["x_2"], q["y_2"], ties=True).pvalue == pytest.approx(
        0.0010040217
    )

    assert xicorr(q["x_4"], q["y_4"], ties=True).pvalue == pytest.approx(
        0.1838021283
    )
    assert xicorr(q["x_3"], q["y_3"], ties=True).pvalue == pytest.approx(
        0.001986300
    )


def test_errors():
    with pytest.raises(ValueError):
        xicorr([1.0, 2.0, 3.0, 4.0], [5.0, 6.0])

    with pytest.raises(ValueError):
        xicorr(np.array([1, 2, 3, 4]), np.array([5, 6]))


def test_flatten():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 4], [9, 16]])
    xi, pvalue = xicorr(x, y)
    assert xi == pytest.approx(0.4)
    assert pvalue == pytest.approx(0.13362874657719392)


def test_empty():
    xi, pvalue = xicorr([], [], ties=False)
    assert xi is np.nan
    assert pvalue is np.nan
