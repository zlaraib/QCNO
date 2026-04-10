import numpy as np

def geometric_func(geometric_name, p_hat, i, j):
    if geometric_name == "none":
        return 1

    elif geometric_name == "physical":
        return 1 - np.dot(p_hat[i], p_hat[j])

    else:
        raise ValueError("geometric_func not implemented")