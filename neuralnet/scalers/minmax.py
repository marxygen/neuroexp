import numpy as np


def min_max_scale(data: np.array):
    return ((data - data.mean()) / (data.max() - data.mean())).astype(np.longfloat)
