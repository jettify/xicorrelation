xicorrelation
=============
.. image:: https://github.com/jettify/xicorrelation/workflows/CI/badge.svg
   :target: https://github.com/jettify/xicorrelation/actions?query=workflow%3ACI
   :alt: GitHub Actions status for master branch
.. image:: https://codecov.io/gh/jettify/xicorrelation/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jettify/xicorrelation
.. image:: https://img.shields.io/pypi/pyversions/xicorrelation.svg
    :target: https://pypi.org/project/xicorrelation
.. image:: https://img.shields.io/pypi/v/xicorrelation.svg
    :target: https://pypi.python.org/pypi/xicorrelation
..
.. image:: https://readthedocs.org/projects/xicorrelation/badge/?version=latest
    :target: https://xicorrelation.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


**xicorrelation** package implements xi correlation formula proposed in  https://arxiv.org/pdf/1909.10140.pdf.


It is based off the R code mentioned in the paper: https://statweb.stanford.edu/~souravc/xi.R and
R package https://github.com/cran/XICOR

+-----------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/jettify/xicorrelation/master/docs/anscombe.png   |
+-----------------------------------------------------------------------------------------------+


Simple example
--------------

.. code:: python

    from xicorrelation import xicorr

    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

    # API similar to spearmanr or kendalltau from scipy
    xi, pvalue = xicorr(x, y)
    print("xi", xi)
    print("pvalue", pvalue)


Installation
------------
Installation process is simple, just::

    $ pip install xicorrelation
