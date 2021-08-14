import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


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
