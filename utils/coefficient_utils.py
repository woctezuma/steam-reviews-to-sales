import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, RepeatedKFold


def get_estimator_coefs_for_cv(est, cv_model):
    estimator_name = list(cv_model["estimator"][0].named_steps.keys())[-1]

    try:
        estimator_coefs = est.named_steps[estimator_name].regressor_.coef_
    except AttributeError:
        estimator_coefs = est.named_steps[estimator_name].coef_

    return estimator_coefs


def cross_validate_model(model, X, y):
    # Reference:
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html

    cv_model = cross_validate(
        model,
        X,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=5),
        return_estimator=True,
        n_jobs=-1,
    )

    scales = X.std(axis=0)

    coefs = pd.DataFrame(
        [
            get_estimator_coefs_for_cv(est, cv_model) * scales
            for est in cv_model["estimator"]
        ],
        columns=X.columns,
    )

    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs, orient="h", color="k", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title("Coefficient importance and its variability")
    plt.subplots_adjust(left=0.3)
    plt.show()

    return
