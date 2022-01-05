from xicorrelation import xicorr

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

# API similar to spearmanr or kendalltau from scipy
xi, pvalue = xicorr(x, y)
print("xi", xi)
print("pvalue", pvalue)
