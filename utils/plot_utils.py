import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_pie(df, percentage_threshold=5):
    # Reference: https://stackoverflow.com/a/63688022/376454

    def label_function(val):
        n = val / 100 * len(df)
        if val < percentage_threshold:
            s = ""
        else:
            s = f"{val:.0f}% ({n:.0f})"
        return s

    fig, ax = plt.subplots()
    df.groupby("review_score_desc").size().sort_values().plot(
        kind="pie", autopct=label_function, textprops={"fontsize": 12}, ax=ax
    )
    ax.set_ylabel("", size=10)
    plt.tight_layout()
    plt.show()

    return


def plot_df(df, use_log_log_scale=False, x_column_name="total_reviews"):
    fig, ax = plt.subplots()

    if use_log_log_scale:
        # Reference: https://stackoverflow.com/a/23918246/376454
        ax.set(xscale="log", yscale="log")

    # Each datapoint is colored based on its review score (an integer between 0 and 9).
    # Reference: https://stackoverflow.com/a/14887119/376454
    sns.scatterplot(data=df, x=x_column_name, y="sales", ax=ax, hue="review_score")

    return ax


def plot_arrays(x, y, xlabel="#reviews", ylabel="#owners"):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, alpha=0.5, label="observation")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return ax


def superimpose_vginsights(ax, x_test):
    # In a graph titled "Review Multipliers to Use for Units Sold Estimate (Boxleiter ratio) for 2021", VG Insights
    # suggests a range of ratios for different periods of time: pre-2014, 2014-2016, 2017, 2018-2019, 2020 and onwards.
    # Reference: https://vginsights.com/insights/article/how-to-estimate-steam-video-game-sales
    #
    # Below, we use the most lenient, i.e. the largest, range of ratios for the period up to the data leak (July 2018).
    # Ideally, data points should lie between the 2 red lines corresponding to these ratios. In practice, they do not.
    ratio_range = [25, 100]

    for ratio in ratio_range:
        ymean = ratio * x_test
        ax.plot(x_test, ymean, color="red")

    return


def easy_plot(
    df,
    use_log_log_scale=False,
    enforce_plot_limits=False,
    x_column_name="total_reviews",
):
    if use_log_log_scale:
        y_lim = [10, 10 ** 7]
        x_lim = [10, 10 ** 5]
    else:
        y_lim = [0, 2.6 * 10 ** 6]
        x_lim = [0, 1.3 * 10 ** 4]

    ax = plot_df(df, use_log_log_scale=use_log_log_scale, x_column_name=x_column_name)
    superimpose_vginsights(ax, x_test=df[x_column_name])

    if enforce_plot_limits:
        plt.ylim(y_lim)
        plt.xlim(x_lim)

    plt.show()

    return


def plot_predictions(x_train, y_train, x_test, ymean, ystd=None, xlim=None, ylim=None):
    ax = plot_arrays(x_train, y_train)
    ax.plot(x_test, ymean, color="red", label="predict mean")
    if ystd is not None:
        ax.fill_between(
            x_test,
            ymean - ystd,
            ymean + ystd,
            color="pink",
            alpha=0.5,
            label="predict std",
        )
    ax.legend()
    if xlim is not None:
        ax.set_xlim([0, xlim])
    if ylim is not None:
        ax.set_ylim([0, ylim])

    return


def grid_plot(df, x_label="total_reviews", y_label="sales", alpha=0.5):
    x = df[x_label]
    y = df[y_label]

    sns.pairplot(df[[y_label, x_label]], kind="reg", diag_kind="kde")
    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].scatter(x, y, alpha=alpha)
    ax[1, 0].scatter(np.log1p(x), y, alpha=alpha)
    ax[0, 1].scatter(x, np.log1p(y), alpha=alpha)
    ax[1, 1].scatter(np.log1p(x), np.log1p(y), alpha=alpha)

    ax[0, 0].set_xlabel(f"{x_label}")
    ax[1, 0].set_xlabel(f"log({x_label})")
    ax[0, 1].set_xlabel(f"{x_label}")
    ax[1, 1].set_xlabel(f"log({x_label})")

    ax[0, 0].set_ylabel(f"{y_label}")
    ax[1, 0].set_ylabel(f"{y_label}")
    ax[0, 1].set_ylabel(f"log({y_label})")
    ax[1, 1].set_ylabel(f"log({y_label})")

    plt.minorticks_off()
    plt.show()

    return


def pairplot_features(
    df,
    log_plot=True,
    x_label="total_negative",
    y_label="total_positive",
    hue_label="review_score",
    alpha=0.75,
):
    x = df[x_label]
    y = df[y_label]

    if log_plot:
        x = np.log1p(x)
        y = np.log1p(y)

    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue=df[hue_label], alpha=alpha)
    plt.show()

    return


def main():
    matplotlib.use("Qt5Agg")

    return True


if __name__ == "__main__":
    main()
