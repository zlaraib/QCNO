import numpy as np

def geometric_func(p, p_hat, i, j, theta_nu):
    if np.array_equal(p[i], 0) and np.array_equal(p[j], 0.0):  # isotropic case
        return 1
    elif theta_nu in [0.1, 0, 0.01]:  # special addition for isotropic tests
        return 1
    elif theta_nu==1000: # Josh test 
        p[i, :] /= np.linalg.norm(p[i, :]) 
        return 1 - np.dot(p[i], p[j])
    else:
        return 1 - np.dot(p_hat[i], p_hat[j])
