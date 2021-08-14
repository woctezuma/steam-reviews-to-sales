import tidfit
from matplotlib import pyplot as plt

from utils.test_utils import check_test_apps
from utils.benchmark_utils import benchmark_models
from utils.coefficient_utils import cross_validate_model
from utils.plot_utils import plot_predictions, plot_arrays, pairplot_features
from utils.regression_utils import fit_linear_model


def run_1d_fit(
    df,
    fit_intercept=True,
    standardize_input=False,
    apply_ransac=False,
    apply_log_to_target=False,
    apply_log_to_input=False,
    num_segments_pwl=3,
    verbose=True,
):
    ## Single feature
    features = ["total_reviews"]

    X = df[features]
    y = df["sales"]

    model = fit_linear_model(
        X,
        y,
        fit_intercept=fit_intercept,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_log_to_target=apply_log_to_target,
    )
    check_test_apps(model, features)

    benchmark_models(
        X,
        y,
        fit_intercept=fit_intercept,
        apply_log_to_input=apply_log_to_input,
        apply_log_to_target=apply_log_to_target,
        num_segments_pwl=num_segments_pwl,
    )

    plot_predictions(X, y, X.squeeze(), model.predict(X))
    plt.show()

    x_train = X.squeeze()
    y_train = y

    # Reference: https://github.com/aminnj/tidfit

    if verbose:
        plot_arrays(x_train, y_train)
        out = tidfit.fit("a*x", x_train, y_train)
        plt.show()

    if verbose:
        plot_arrays(x_train, y_train)
        out = tidfit.fit("a*x+b", x_train, y_train)
        plt.show()

    return


def run_2d_fit(
    df,
    fit_intercept=True,
    standardize_input=False,
    apply_ransac=False,
    apply_log_to_target=False,
    verbose=True,
):
    ## Two features
    features = ["total_positive", "total_negative"]

    X = df[features]
    y = df["sales"]

    if verbose:
        pairplot_features(df, log_plot=True)

    model = fit_linear_model(
        X,
        y,
        fit_intercept=fit_intercept,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_log_to_target=apply_log_to_target,
    )
    check_test_apps(model, features)

    cross_validate_model(model, X, y)

    return
