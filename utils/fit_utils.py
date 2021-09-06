from matplotlib import pyplot as plt

from utils.benchmark_utils import benchmark_models
from utils.coefficient_utils import cross_validate_model
from utils.mapie_utils import predict_with_mapie
from utils.plot_utils import plot_predictions, pairplot_features
from utils.regression_utils import fit_linear_model
from utils.test_utils import check_test_apps
from utils.tidfit_utils import run_linear_tidfit, run_chance_tidfit
from utils.transform_utils import (
    get_review_chance,
    get_review_multiplier,
    invert_review_chance,
    invert_review_multiplier,
)


def run_1d_fit(
    df,
    apply_train_test_split=True,
    fit_intercept=True,
    specific_base_estimator_name="",
    standardize_input=False,
    apply_ransac=False,
    apply_mapie=False,
    apply_log_to_target=False,
    apply_log_to_input=False,
    num_segments_pwl=3,
    features=None,
    target_name=None,
    verbose=True,
):
    ## Single feature
    if features is None:
        features = ["total_reviews"]

    if target_name is None:
        target_name = "sales"

    X = df[features]
    y = df[target_name]

    model = fit_linear_model(
        X,
        y,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        specific_base_estimator_name=specific_base_estimator_name,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_mapie=apply_mapie,
        apply_log_to_target=apply_log_to_target,
    )
    check_test_apps(model, features)

    pwl_curve = benchmark_models(
        X,
        y,
        fit_intercept=fit_intercept,
        apply_log_to_input=apply_log_to_input,
        apply_log_to_target=apply_log_to_target,
        num_segments_pwl=num_segments_pwl,
    )
    print("Piece-wise Linear Model predictions:")
    check_test_apps(pwl_curve, features)

    ypred, ystd = predict_with_mapie(model, X)
    plot_predictions(X, y, X.squeeze(), ypred, ystd)
    plt.show()

    if verbose:
        run_linear_tidfit(X, y)

    return


def run_2d_fit(
    df,
    apply_train_test_split=True,
    fit_intercept=True,
    specific_base_estimator_name="",
    standardize_input=False,
    apply_ransac=False,
    apply_mapie=False,
    apply_log_to_target=False,
    features=None,
    target_name=None,
    verbose=True,
):
    ## Two features
    if features is None:
        features = ["total_positive", "total_negative"]

    if target_name is None:
        target_name = "sales"

    X = df[features]
    y = df[target_name]

    if verbose:
        pairplot_features(df, log_plot=True)

    model = fit_linear_model(
        X,
        y,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        specific_base_estimator_name=specific_base_estimator_name,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_mapie=apply_mapie,
        apply_log_to_target=apply_log_to_target,
    )
    check_test_apps(model, features)

    if specific_base_estimator_name not in ["boosting"]:
        cross_validate_model(model, X, y)

    return


def run_1d_fit_to_chance(
    df,
    apply_train_test_split=True,
    fit_intercept=True,
    specific_base_estimator_name="",
    apply_mapie=False,
    chance_target="review_chance",
    verbose=True,
):
    ## Single feature
    features = ["total_reviews"]

    # A list instead of a str for consistency.
    # Caveat: done differently in run_1d_fit() and run_2d_fit(), where this variable was a str.
    target = ["sales"]

    X = df[features]
    y = df[target]

    if chance_target == "review_chance":
        y_new_target = get_review_chance(X, y)
        inverse_func = invert_review_chance
        ylabel = "chance to write a review"
        ylim = 0.09
    else:
        y_new_target = get_review_multiplier(X, y)
        inverse_func = invert_review_multiplier
        ylabel = "review multiplier"
        ylim = 700

    model = fit_linear_model(
        X,
        y_new_target,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        standardize_input=False,
        apply_ransac=False,
        apply_mapie=apply_mapie,
        apply_log_to_target=False,
        specific_base_estimator_name=specific_base_estimator_name,
    )

    check_test_apps(model, features, transform_output=True, inverse_func=inverse_func)

    ypred, ystd = predict_with_mapie(model, X)

    y_old_pred = inverse_func(X.squeeze(), ypred)

    if verbose:
        plot_predictions(
            X,
            y_new_target,
            X.squeeze(),
            ypred.squeeze(),
            ystd,
            ylim=ylim,
            ylabel=ylabel,
        )
        plt.xscale("log")
        plt.show()

    if verbose:
        plot_predictions(X, y, X.squeeze(), y_old_pred, ylabel="#owners")
        plt.loglog()
        plt.show()

    if verbose:
        # Caveat: do not use y_new_target here, since transform is already taken care of by the formula input to tidfit.
        run_chance_tidfit(X, y, ylabel="#owners")

    return
