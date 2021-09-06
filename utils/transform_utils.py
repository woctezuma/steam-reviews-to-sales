import numpy as np


def get_review_chance(X, y):
    # Inputs:
    # - X is the number of reviews
    # - y is the number of sales
    # Output:
    # - chance to write a review
    return np.divide(X, np.asarray(y))


def invert_review_chance(X, chance):
    # Inputs:
    # - X is the number of reviews
    # - c is the chance to write a review
    # Output:
    # - number of sales
    return np.divide(X, np.asarray(chance))


def get_review_multiplier(X, y):
    # Inputs:
    # - X is the number of reviews
    # - y is the number of sales
    # Output:
    # - "review multiplier"
    return np.divide(np.asarray(y), (1 + X))


def invert_review_multiplier(X, multiplier):
    # Inputs:
    # - X is the number of reviews
    # - m is the "review multiplier"
    # Output:
    # - number of sales
    return np.multiply(np.asarray(multiplier), (1 + X))
