import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt


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


def plot_predictions(x_train, y_train, x_test, ymean, ystd=0):
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

    return


def main():
    matplotlib.use("Qt5Agg")

    return True


if __name__ == "__main__":
    main()
