import numpy as np
from mapie.estimators import MapieRegressor
from sklearn import linear_model, preprocessing, pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split


def fit_linear_model(
    X,
    y,
    apply_train_test_split=True,
    fit_intercept=True,
    standardize_input=False,
    apply_ransac=False,
    apply_mapie=False,
    apply_log_to_target=False,
    ransac_fraction=0.1,
    verbose=True,
):
    # Reference: https://scikit-learn.org/stable/supervised_learning.html

    if apply_train_test_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    scaler = preprocessing.StandardScaler(
        with_mean=standardize_input, with_std=standardize_input
    )

    log_scaler = preprocessing.FunctionTransformer(func=np.log1p, validate=True)
    poly_scaler = preprocessing.PolynomialFeatures(degree=3, include_bias=False)

    base_estimator = linear_model.LinearRegression(fit_intercept=fit_intercept)
    base_estimator = linear_model.BayesianRidge(fit_intercept=fit_intercept)
    base_estimator = linear_model.RidgeCV(fit_intercept=fit_intercept)

    if verbose:
        print(base_estimator)

    ransac_estimator = linear_model.RANSACRegressor(
        base_estimator=base_estimator,
        min_samples=int(ransac_fraction * X.shape[0]),
        max_trials=1e3,
    )

    if apply_ransac:
        estimator = ransac_estimator
    else:
        estimator = base_estimator

    if apply_log_to_target:
        estimator = TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )

    pipe = pipeline.make_pipeline(scaler, estimator)

    if apply_mapie:
        model = MapieRegressor(pipe)
    else:
        model = pipe

    model.fit(X_train, y_train)

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    print(f"[training] R²: {score_train}")
    print(f"[test] R²: {score_test}")

    return model
