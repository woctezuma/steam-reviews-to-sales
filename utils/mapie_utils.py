import numpy as np

from utils.transform_utils import invert_review_chance


def get_alpha():
    # Alpha for 1 and 2 standard-deviations
    alpha = [0.05, 0.32]
    return alpha


def get_alpha_index():
    # Arbitrarily chosen so that alpha = 0.32. cf. get_alpha()
    alpha_index_to_display = 1
    return alpha_index_to_display


def predict_with_mapie(model, X):
    alpha = get_alpha()
    alpha_index_to_display = get_alpha_index()

    try:
        ypred, ypis = model.predict(X, alpha=alpha)

        lower_bound = ypis[:, 0, alpha_index_to_display]
        upper_bound = ypis[:, 1, alpha_index_to_display]
        ystd = 0.5 * (upper_bound - lower_bound).ravel()

    except TypeError:
        try:
            ypred, ystd = model.predict(X, return_std=True)

        except TypeError:
            ypred = model.predict(X)
            ystd = 0

    return ypred, ystd


def get_prediction_bounds(ypred, ystd, required_to_be_positive=True):
    p_lower = ypred - ystd
    p_upper = ypred + ystd

    if required_to_be_positive:
        machine_epsilon = np.finfo(float).eps

        p_lower = max(machine_epsilon, p_lower)
        p_upper = max(machine_epsilon, p_upper)

    return p_lower, p_upper


def invert_prediction_bounds(X, p_lower, p_upper):
    inv_p_lower = invert_review_chance(X, p_lower)
    inv_p_upper = invert_review_chance(X, p_upper)

    p_lower = min(inv_p_lower, inv_p_upper)
    p_upper = max(inv_p_lower, inv_p_upper)

    return p_lower, p_upper
