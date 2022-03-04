import numpy as np


def generate_polynomial_dataset(min_=-30, max_=30):
    X = np.arange(min_, max_, 1).reshape(len)
    y = 9*X**3 + 5*X**2 + np.random.randn(60)*1000
    return X, y
