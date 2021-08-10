import matplotlib
import matplotlib.pyplot as plt
import tidfit
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from analyze_data import (
    load_aggregated_data_as_df,
    remove_extreme_values,
)
from utils.plot_utils import plot_arrays, plot_predictions, plot_pie
from utils.reviews_utils import unify_descriptions


def fit_linear_model(X, y, fit_intercept=True):
    # Reference: https://scikit-learn.org/stable/supervised_learning.html

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    print(f"[training] R²: {score_train}")
    print(f"[test] R²: {score_test}")
    print(f"[model] coefficients: {model.coef_}")
    print(f"[model] intercept: {model.intercept_}")

    return model


def main():
    matplotlib.use("Qt5Agg")

    df = load_aggregated_data_as_df(sort_by_num_reviews=True)
    plot_pie(df)
    plot_pie(unify_descriptions(df), percentage_threshold=1)
    df = remove_extreme_values(df, "sales")
    df = remove_extreme_values(df, "total_reviews", 0.25, 1.0)

    # Filter by review score
    df = df[df["review_score"] == 9]

    X = df.loc[:, df.columns == "total_reviews"]
    y = df["sales"]

    x_train = X.squeeze()
    y_train = y

    # Reference: https://github.com/aminnj/tidfit

    plot_arrays(x_train, y_train)
    out = tidfit.fit("a*x", x_train, y_train)
    plt.show()

    plot_arrays(x_train, y_train)
    out = tidfit.fit("a*x+b", x_train, y_train)
    plt.show()

    model = fit_linear_model(X, y, fit_intercept=True)
    y_pred = model.predict(X)
    plot_predictions(X, y, X.squeeze(), ymean=y_pred, ystd=0)
    plt.show()

    return True


if __name__ == "__main__":
    main()
