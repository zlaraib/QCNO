# rho_from_counts.py
#
# Post-processing: turn the sampled single-site Pauli sigma files written by the
# time-evolution driver into single-site density-matrix components. These are
# algebraic functions of the sigmas, so they are computed once after the run from
# t_sigma_{x,y,z}.dat rather than inside the evolution loop.

import os

import numpy as np


def rho_from_counts(datadir):
    """
    Compute single-site density-matrix components from the sampled sigma files.

    Reads t_sigma_x.dat, t_sigma_y.dat, t_sigma_z.dat from datadir. Each file has
    one row per timestep: a leading time column followed by the per-site
    <sigma_b> in original particle order. Per site:

        rho_ee   = (sigma_z + 1) / 2      (electron-flavor probability)
        rho_mumu = (1 - sigma_z) / 2      (mu-flavor probability)
        rho_emu  = 0.5 * sqrt(sigma_x^2 + sigma_y^2)   (coherence magnitude)

    Writes t_rho_ee_from_counts.dat, t_rho_mumu_from_counts.dat, and
    t_rho_emu_from_counts.dat (time column + per-site values, matching the sigma
    file layout).

    Inputs:
    - datadir: directory holding the t_sigma_{x,y,z}.dat files; the rho files are
      written back into it.

    Output:
    - (times, rho_ee, rho_mumu, rho_emu): the time array and the three per-site
      component arrays (shape [n_times, n_sites]).
    """
    def load(basis):
        return np.atleast_2d(np.loadtxt(os.path.join(datadir, f"t_sigma_{basis}.dat")))

    sx_data = load("x")
    sy_data = load("y")
    sz_data = load("z")

    if not (sx_data.shape == sy_data.shape == sz_data.shape):
        raise ValueError(
            "t_sigma_{x,y,z}.dat have mismatched shapes "
            f"({sx_data.shape}, {sy_data.shape}, {sz_data.shape}); "
            "they must share the same times and sites."
        )

    times = sz_data[:, 0]
    sx = sx_data[:, 1:]
    sy = sy_data[:, 1:]
    sz = sz_data[:, 1:]

    rho_ee = (sz + 1.0) / 2.0
    rho_mumu = (1.0 - sz) / 2.0
    rho_emu = 0.5 * np.sqrt(sx**2 + sy**2)

    for fname, rho in (
        ("t_rho_ee_from_counts.dat", rho_ee),
        ("t_rho_mumu_from_counts.dat", rho_mumu),
        ("t_rho_emu_from_counts.dat", rho_emu),
    ):
        np.savetxt(os.path.join(datadir, fname), np.column_stack([times, rho]), fmt="%.16e")

    return times, rho_ee, rho_mumu, rho_emu
