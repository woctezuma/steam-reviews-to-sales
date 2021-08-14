import matplotlib

from analyze_data import (
    load_aggregated_data_as_df,
)
from utils.fit_utils import run_1d_fit, run_2d_fit
from utils.outlier_utils import detect_outliers
from utils.plot_utils import plot_pie, grid_plot
from utils.reviews_utils import unify_descriptions


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
