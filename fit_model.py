import matplotlib
import seaborn as sns

from analyze_data import (
    load_aggregated_data_as_df,
)
from utils.fit_utils import run_1d_fit, run_1d_fit_to_chance, run_2d_fit
from utils.outlier_utils import detect_outliers
from utils.plot_utils import grid_plot, plot_boxleiter_ratios, plot_pie
from utils.reviews_utils import unify_descriptions


def main():
    matplotlib.use("Qt5Agg")

    # Reference: https://seaborn.pydata.org/tutorial/color_palettes.html#qualitative-color-palettes
    sns.set_palette(sns.color_palette("colorblind"))

    apply_train_test_split = False
    fit_intercept = True
    specific_base_estimator_name = ""
    standardize_input = False
    apply_ransac = False
    apply_mapie = True
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

    # One of:
    # - review_chance
    # - review_multiplier
    chance_target = "review_chance"

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

    run_1d_fit_to_chance(
        df,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        specific_base_estimator_name=specific_base_estimator_name,
        apply_mapie=apply_mapie,
        chance_target=chance_target,
        verbose=verbose,
    )

    if verbose:
        grid_plot(df)
        plot_boxleiter_ratios(df)

    run_1d_fit(
        df,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        specific_base_estimator_name=specific_base_estimator_name,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_mapie=apply_mapie,
        apply_log_to_target=apply_log_to_target,
        apply_log_to_input=apply_log_to_input,
        num_segments_pwl=num_segments_pwl,
        features=["total_reviews"],
        target_name="sales",
        verbose=verbose,
    )

    run_2d_fit(
        df,
        apply_train_test_split=apply_train_test_split,
        fit_intercept=fit_intercept,
        specific_base_estimator_name=specific_base_estimator_name,
        standardize_input=standardize_input,
        apply_ransac=apply_ransac,
        apply_mapie=False,  # too slow if set to True
        apply_log_to_target=apply_log_to_target,
        features=["total_positive", "total_negative"],
        target_name="sales",
        verbose=verbose,
    )

    return True


if __name__ == "__main__":
    main()
