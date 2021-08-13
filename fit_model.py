import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tidfit

from analyze_data import (
    load_aggregated_data_as_df,
)
from utils.plot_utils import (
    plot_arrays,
    plot_predictions,
    plot_pie,
    grid_plot,
    pairplot_features,
)
from utils.regression_utils import (
    benchmark_models,
    fit_linear_model,
    detect_outliers,
    cross_validate_model,
)
from utils.reviews_utils import unify_descriptions


def get_test_apps():
    all_features = ["review_score", "total_positive", "total_negative", "total_reviews"]

    # Reference: https://store.steampowered.com/appreviews/892970?json=1&num_per_page=0&language=all&purchase_type=all&filter_offtopic_activity=0
    test_apps = {
        "Valheim": {"appID": 892970, "data": [9, 246944, 10706, 257650]},
        "Iratus": {"appID": 807120, "data": [8, 4606, 698, 5304]},
    }

    return test_apps, all_features


def check_test_apps(model, features):
    test_apps, all_features = get_test_apps()

    for app_name in test_apps.keys():
        v_test = test_apps[app_name]["data"]

        v = [v_test[i] for (i, f) in enumerate(all_features) if f in features]
        v = np.array(v).reshape(1, -1)

        p = model.predict(v)

        print(f"Input: {app_name} ---> predicted sales: {p[0] / 1e6:.3f} M")

    return


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


def main():
    matplotlib.use("Qt5Agg")

    fit_intercept = True
    standardize_input = False
    apply_ransac = False
    apply_log_to_target = False
    apply_log_to_input = False
    num_segments_pwl = 3
    verbose = False

    # If a game has fewer than 500 reviews, I am not sure that we could infer a lot anyway, because:
    # - there is a lot of variability in the range [0, 500] reviews,
    # - games cannot have "overwhelmingly" bad or good ratings (review_score 1 or 9).
    # NB: With this threshold, games with review_score 0, 3 or 7 are necessarily excluded!
    threshold_num_reviews = 5e2

    # Caveat: the following choice is extremely important, because outliers impact the regression a lot!
    # Typical choices would be:
    # - either "minkowski" to remove games with many reviews, or bad ratings (data bias towards positive ratings)
    # - or "cosine" to remove games which do not align well with others. This is very relevant for linear regression!
    outlier_metric = "cosine"

    df = load_aggregated_data_as_df(sort_by_num_reviews=True)
    if verbose:
        plot_pie(df)
        plot_pie(unify_descriptions(df), percentage_threshold=1)

    df = df.drop(["name", "review_score_desc"], axis=1)

    # Filter out data points with few reviews
    df = df[df["total_reviews"] >= threshold_num_reviews]

    # Filter out outliers (based on LocalOutlierFactor)
    is_inlier = detect_outliers(df, metric=outlier_metric, verbose=verbose)
    df = df[is_inlier]

    if verbose:
        grid_plot(df)

    run_1d_fit(
        df,
        fit_intercept=fit_intercept,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_log_to_target=apply_log_to_target,
        apply_log_to_input=apply_log_to_input,
        num_segments_pwl=num_segments_pwl,
        verbose=verbose,
    )

    run_2d_fit(
        df,
        fit_intercept=fit_intercept,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_log_to_target=apply_log_to_target,
        verbose=verbose,
    )

    return True


if __name__ == "__main__":
    main()
