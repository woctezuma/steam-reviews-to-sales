import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_data import load_output_dict
from utils.data_utils import load_input_data


def load_aggregated_data_as_dict(verbose=True):
    player_data = load_input_data()
    review_data = load_output_dict()

    app_ids = set(player_data.keys()).intersection(review_data.keys())
    app_ids = sorted(app_ids, key=int)

    data = dict()
    for app_id in app_ids:
        data[app_id] = dict()

        for key in player_data[app_id].keys():
            data[app_id][key] = player_data[app_id][key]

        for key in review_data[app_id].keys():
            data[app_id][key] = review_data[app_id][key]

    if verbose:
        print(f"#apps = {len(data)}")

    return data


def load_aggregated_data_as_df(sort_by_num_reviews=False, verbose=True):
    data = load_aggregated_data_as_dict(verbose=verbose)
    df = pd.DataFrame.from_dict(data, orient="index")
    if sort_by_num_reviews:
        df = df.sort_values("total_reviews")

    # Ensure that review_score is interpreted as a categorical variable
    df["review_score"] = df["review_score"].astype("category")

    if verbose:
        print(df[["total_reviews", "sales"]].describe())
        print(list(df.columns))

    return df


def remove_extreme_values(
    df,
    column_name,
    q_min=0.01,
    q_max=0.99,
    verbose=True,
):
    threshold_min = df[column_name].quantile(q_min)
    threshold_max = df[column_name].quantile(q_max)

    num_rows = len(df)
    df = df[df[column_name] >= threshold_min]
    df = df[df[column_name] <= threshold_max]

    if verbose:
        print(f"[filtering {column_name}] #rows: {num_rows} ---> {len(df)}")

    return df


def get_arrays_from(df):
    x_train = np.array(df["total_reviews"])
    y_train = np.array(df["sales"])

    return x_train, y_train


def plot_data(x, y, xlabel="#reviews", ylabel="#owners"):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, alpha=0.5, label="observation")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return ax


def main():
    matplotlib.use("Qt5Agg")

    df = load_aggregated_data_as_df()
    df = remove_extreme_values(df, "sales")
    df = remove_extreme_values(df, "total_reviews")
    x_train, y_train = get_arrays_from(df)

    plot_data(x_train, y_train, xlabel="#reviews", ylabel="#owners")
    plt.show()

    df = load_aggregated_data_as_df()
    df = remove_extreme_values(df, "sales")
    df = remove_extreme_values(df, "total_reviews", 0.25, 1.0)
    x_train, y_train = get_arrays_from(df)

    y_train = np.log10(y_train)
    x_train = np.log10(x_train)

    plot_data(x_train, y_train, xlabel="log10(#reviews)", ylabel="log10(#owners)")
    plt.show()

    return True


if __name__ == "__main__":
    main()
