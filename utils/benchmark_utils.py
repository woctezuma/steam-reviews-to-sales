from time import time

import numpy as np
from matplotlib import pyplot as plt
from pwlfit import fitter
from scipy import stats
from scipy.optimize import least_squares
from sklearn import linear_model, svm


def describe_pwlfit(curve):
    print("Knots:")
    print(curve)

    print("Slopes:")
    print(
        [
            (j[1] - i[1]) / (j[0] - i[0])
            for (i, j) in zip(curve.points[:-1], curve.points[1:])
        ],
    )

    return


def benchmark_models(
    X,
    y,
    fit_intercept=True,
    apply_log_to_input=False,
    apply_log_to_target=False,
    num_segments_pwl=1,
):
    if apply_log_to_input:
        X = np.log1p(X)

    if apply_log_to_target:
        y = np.log1p(y)

    # Reference: https://scipy-cookbook.readthedocs.io/items/robust_regression.html
    if fit_intercept:
        x0 = [1, 1]

        def f(p, x_train, y_train):
            return p[0] * x_train + p[1] - y_train

    else:
        x0 = 1

        def f(p, x_train, y_train):
            return p[0] * x_train - y_train

    est_reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
    est_svm = svm.LinearSVR(fit_intercept=fit_intercept)

    tic = time()
    res = stats.linregress(
        X.squeeze(),
        y,
    )  # same as LinearRegression (necessarily with intercept)
    res_lsq = least_squares(
        f,
        x0=x0,
        loss="linear",
        args=(X.squeeze(), y),
    )  # same as LinearRegression
    res_robust = least_squares(
        f,
        x0=x0,
        loss="soft_l1",
        f_scale=0.1,
        args=(X.squeeze(), y),
    )  # roughly the same as Huber
    res_huber = least_squares(
        f,
        x0=x0,
        loss="huber",
        f_scale=0.1,
        args=(X.squeeze(), y),
    )
    res_cauchy = least_squares(
        f,
        x0=x0,
        loss="cauchy",
        f_scale=0.1,
        args=(X.squeeze(), y),
    )
    res_arctan = least_squares(
        f,
        x0=x0,
        loss="arctan",
        f_scale=0.1,
        args=(X.squeeze(), y),
    )
    print(f"done in {time() - tic:.3f}s")

    tic = time()
    est_reg.fit(X, y)
    est_svm.fit(X, y)
    print(f"done in {time() - tic:.3f}s")

    tic = time()
    curve = fitter.fit_pwl(X.squeeze(), y, num_segments=num_segments_pwl)
    print(f"done in {time() - tic:.3f}s")
    describe_pwlfit(curve)

    fig, ax = plt.subplots()

    _ = ax.plot(
        X,
        res_cauchy.x[0] * X + res_cauchy.x[1],
        "orange",
        label=f"SciPy (cauchy {res_cauchy.x[0]:.0f})",
    )

    _ = ax.plot(
        X,
        est_svm.predict(X),
        "blue",
        label=f"LinearSVR {est_svm.coef_[0]:.0f}",
    )

    _ = ax.plot(
        X,
        res_huber.x[0] * X + res_huber.x[1],
        "red",
        label=f"SciPy (huber {res_huber.x[0]:.0f})",
    )

    _ = ax.plot(
        X,
        est_reg.predict(X),
        "green",
        label=f"LinearRegression {est_reg.coef_[0]:.0f}",
    )

    _ = ax.plot(
        X,
        res_arctan.x[0] * X + res_arctan.x[1],
        "yellow",
        label=f"SciPy (arctan {res_arctan.x[0]:.0f})",
    )

    _ = ax.plot(X, curve.predict(X), "black", label="pwlfit")

    _ = ax.scatter(X, y, alpha=0.25)

    if apply_log_to_input:
        ax.set_xlabel("log(#reviews)")
    else:
        ax.set_xlabel("#reviews")

    if apply_log_to_target:
        ax.set_ylabel("log(#sales)")
    else:
        ax.set_ylabel("#sales")

    ax.legend()
    plt.show()

    return curve
