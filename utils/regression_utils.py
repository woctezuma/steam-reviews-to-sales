from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pwlfit import fitter
from scipy import stats
from scipy.optimize import least_squares
from sklearn import linear_model, svm, preprocessing, pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def fit_linear_model(
        X,
        y,
        fit_intercept=True,
        standardize_input=False,
        apply_ransac=False,
        apply_log_to_target=False,
        ransac_fraction=0.1,
):
    # Reference: https://scikit-learn.org/stable/supervised_learning.html

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = preprocessing.StandardScaler(
        with_mean=standardize_input, with_std=standardize_input
    )

    log_scaler = preprocessing.FunctionTransformer(func=np.log1p, validate=True)
    poly_scaler = preprocessing.PolynomialFeatures(degree=3, include_bias=False)

    base_estimator = linear_model.LinearRegression(fit_intercept=fit_intercept)
    base_estimator = linear_model.BayesianRidge(fit_intercept=fit_intercept)
    base_estimator = linear_model.RidgeCV(fit_intercept=fit_intercept)

    ransac_estimator = linear_model.RANSACRegressor(
        base_estimator=base_estimator,
        min_samples=int(ransac_fraction * X.shape[0]),
        max_trials=1e3,
    )

    if apply_ransac:
        estimator = ransac_estimator
    else:
        estimator = base_estimator

    if apply_log_to_target:
        estimator = TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )

    model = pipeline.make_pipeline(scaler, estimator)

    model.fit(X_train, y_train)

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    print(f"[training] R²: {score_train}")
    print(f"[test] R²: {score_test}")

    return model


def get_estimator_coefs_for_cv(est, cv_model):
    estimator_name = list(cv_model["estimator"][0].named_steps.keys())[-1]

    try:
        estimator_coefs = est.named_steps[estimator_name].regressor_.coef_
    except AttributeError:
        estimator_coefs = est.named_steps[estimator_name].coef_

    return estimator_coefs


def cross_validate_model(model, X, y):
    # Reference:
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html

    cv_model = cross_validate(
        model,
        X,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=5),
        return_estimator=True,
        n_jobs=-1,
    )

    scales = X.std(axis=0)

    coefs = pd.DataFrame(
        [
            get_estimator_coefs_for_cv(est, cv_model) * scales
            for est in cv_model["estimator"]
        ],
        columns=X.columns,
    )

    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs, orient="h", color="k", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title("Coefficient importance and its variability")
    plt.subplots_adjust(left=0.3)
    plt.show()

    return


def describe_pwlfit(curve):
    print('Knots:')
    print(curve)

    print('Slopes:')
    print(
        [
            (j[1] - i[1]) / (j[0] - i[0])
            for (i, j) in zip(curve.points[:-1], curve.points[1:])
        ]
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
        X.squeeze(), y
    )  # same as LinearRegression (necessarily with intercept)
    res_lsq = least_squares(
        f, x0=x0, loss="linear", args=(X.squeeze(), y)
    )  # same as LinearRegression
    res_robust = least_squares(
        f, x0=x0, loss="soft_l1", f_scale=0.1, args=(X.squeeze(), y)
    )  # roughly the same as Huber
    res_huber = least_squares(
        f, x0=x0, loss="huber", f_scale=0.1, args=(X.squeeze(), y)
    )
    res_cauchy = least_squares(
        f, x0=x0, loss="cauchy", f_scale=0.1, args=(X.squeeze(), y)
    )
    res_arctan = least_squares(
        f, x0=x0, loss="arctan", f_scale=0.1, args=(X.squeeze(), y)
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
        X, est_svm.predict(X), "blue", label=f"LinearSVR {est_svm.coef_[0]:.0f}"
    )

    _ = ax.plot(
        X,
        res_huber.x[0] * X + res_huber.x[1],
        "red",
        label=f"SciPy (huber {res_huber.x[0]:.0f})",
    )

    _ = ax.plot(
        X, est_reg.predict(X), "green", label=f"LinearRegression {est_reg.coef_[0]:.0f}"
    )

    _ = ax.plot(
        X,
        res_cauchy.x[0] * X + res_cauchy.x[1],
        "orange",
        label=f"SciPy (cauchy {res_cauchy.x[0]:.0f})",
    )

    _ = ax.plot(
        X,
        res_arctan.x[0] * X + res_arctan.x[1],
        "yellow",
        label=f"SciPy (arctan {res_arctan.x[0]:.0f})",
    )

    _ = ax.plot(X, curve.predict(X), "black", label=f"pwlfit")

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

    return


def detect_outliers(
        df,
        features=None,
        n_neighbors=35,
        outliers_fraction="auto",
        metric="minkowski",
        log_plot=True,
        verbose=True,
):
    # Reference: https://scikit-learn.org/stable/modules/outlier_detection.html

    if features is None:
        # Caveat: the order matters for the axes of the plot!
        features = ["total_negative", "total_positive", "total_reviews"]

    X = np.array(df[features])

    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=outliers_fraction, metric=metric
    )
    y_pred = clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_

    if outliers_fraction == "auto":
        outliers_fraction = np.sum(y_pred != 1) / len(y_pred)

    if verbose:
        plt.title(
            f"Local Outlier Factor (LOF) with n={n_neighbors} and c={outliers_fraction:.2f}"
        )
        plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")

        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(
            X[:, 0],
            X[:, 1],
            s=1000 * radius,
            edgecolors="r",
            facecolors="none",
            label="Outlier scores",
        )

        # plot points in color
        colors = np.array(["#377eb8", "#ff7f00"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlabel(features[0])
        plt.ylabel(features[1])

        if log_plot:
            plt.xscale("log")
            plt.yscale("log")

        plt.show()

    is_inlier = y_pred == 1

    return is_inlier
