import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tidfit
from sklearn.linear_model import BayesianRidge

from analyze_data import (
    load_aggregated_data_as_df,
    remove_extreme_values,
    get_arrays_from,
    plot_arrays,
)


def main():
    matplotlib.use("Qt5Agg")

    df = load_aggregated_data_as_df(sort_by_num_reviews=True)
    df = remove_extreme_values(df, "sales")
    df = remove_extreme_values(df, "total_reviews", 0.25, 1.0)

    x_train, y_train = get_arrays_from(df)
    y_train = np.log10(y_train)
    x_train = np.log10(x_train)

    # Reference: https://github.com/aminnj/tidfit

    plot_arrays(x_train, y_train)
    out = tidfit.fit("a*x", x_train, y_train)
    plt.show()

    plot_arrays(x_train, y_train)
    out = tidfit.fit("a*x+b", x_train, y_train)
    plt.show()

    # Reference: https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html

    x_test = x_train

    # Fit by cubic polynomial
    n_order = 3
    X_train = np.vander(x_train, n_order + 1, increasing=True)
    X_test = np.vander(x_test, n_order + 1, increasing=True)

    # Plot the true and predicted curves with log marginal likelihood (L)
    reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    reg.fit(X_train, y_train)
    ymean, ystd = reg.predict(X_test, return_std=True)

    ax = plot_arrays(x_train, y_train)
    ax.plot(x_test, ymean, color="red", label="predict mean")
    ax.fill_between(
        x_test,
        ymean - ystd,
        ymean + ystd,
        color="pink",
        alpha=0.5,
        label="predict std",
    )
    ax.legend()
    plt.show()

    return True


if __name__ == "__main__":
    main()
