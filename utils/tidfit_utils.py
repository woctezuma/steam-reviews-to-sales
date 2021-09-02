import tidfit
from matplotlib import pyplot as plt

from utils.plot_utils import plot_arrays


def run_tidfit(X, y, model_str):
    # Caveat: this is based on the L2 error, and thus is sensitive to the pre-processing of outliers.
    # Reference: https://github.com/aminnj/tidfit

    x_train = X.squeeze()
    y_train = y.squeeze()

    plot_arrays(x_train, y_train)
    out = tidfit.fit(model_str, x_train, y_train)
    plt.show()

    return out


def run_linear_tidfit(X, y):
    # Linear regression, without and with an intercept

    out_a = run_tidfit(X, y, "a*x")
    out_ab = run_tidfit(X, y, "a*x+b")

    return out_a, out_ab


def run_chance_tidfit(X, y):
    # This is equivalent to fitting the chance to write a review: (x/y) ~ a*x+b

    out = run_tidfit(X, y, "x / (a*x+b)")

    return out
