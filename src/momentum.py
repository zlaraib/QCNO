import numpy as np


def vacuum_frequency(delta_m_squared, p, energy_sign):
    """Per-site vacuum oscillation frequency omega = Δm² / (2|p|) * energy_sign.

    Inputs:
    - delta_m_squared: mass-squared difference. Sign/zero conventions are carried
      by the caller through delta_m_squared and energy_sign (e.g. negate
      energy_sign to flip the sign; set delta_m_squared = 0 to disable).
    - p: (N_sites, 3) momentum array.
    - energy_sign: per-site sign (+/-1), one per site.

    Output:
    - omega: (N_sites,) array of vacuum oscillation frequencies.
    """
    p_mod = np.linalg.norm(p, axis=1)
    return (delta_m_squared / (2.0 * p_mod)) * np.asarray(energy_sign)


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
