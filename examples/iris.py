import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import datasets

from xicorrelation import xicorr

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns = iris.feature_names

sns.set(style="ticks", color_codes=True)

g = sns.pairplot(df, palette="husl", markers=["o", "s", "D"])


def corrfunc(x, y, ax=None, **kw):
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y).correlation
    kt = kendalltau(x, y).correlation
    xi = xicorr(x, y).correlation

    ax = ax or plt.gca()
    bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5)
    ax.annotate(
        f"Pearson r = {pr:.2f}",
        xy=(0.1, 0.9),
        xycoords=ax.transAxes,
        bbox=bbox,
        fontsize=7,
    )
    ax.annotate(
        f"Spearman r = {sr:.2f}",
        xy=(0.1, 0.8),
        xycoords=ax.transAxes,
        bbox=bbox,
        fontsize=7,
    )
    ax.annotate(
        f"Kendall tau = {kt:.2f}",
        xy=(0.1, 0.7),
        xycoords=ax.transAxes,
        bbox=bbox,
        fontsize=7,
    )
    ax.annotate(
        f"Chatterjee xi = {xi:.2f}",
        xy=(0.1, 0.6),
        xycoords=ax.transAxes,
        bbox=bbox,
        fontsize=7,
    )


g.map_lower(corrfunc)
g.map_upper(corrfunc)
plt.show()
