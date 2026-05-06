import numpy as np

def momentum(p, N_sites):
    p_hat = []  # Initialize an empty list to collect p_i_hat vectors. This list will contain unit vectors of all sites in all directions
    p_mod = []  # Initialize list that contains the modulus of all sites

    for i in range(N_sites):
        if np.array_equal(p, np.ones((N_sites, 3))):
            p[i, :] /= np.linalg.norm(p[i, :])  # Normalize each site of p to have magnitude 1. Special addition for this omega = pi case
            # Check if this needs to be obeyed for FFI test

        # Calculate the modulus of each site
        p_i_mod = np.linalg.norm(p[i, :])  # The norm returns the magnitude of the vector
        # Calculate unit vector p for each site
        p_i_hat = p[i, :] / p_i_mod

        # Append p_i_hat to the p_hat list
        p_hat.append(p_i_hat)
        # Append p_i_mod to the p_mod list
        p_mod.append(p_i_mod)

    return p_mod, p_hat
