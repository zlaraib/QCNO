# rho_from_counts.py
#
# Post-processing: turn the sampled single-site Pauli sigma files written by the
# time-evolution driver into single-site density-matrix components. These are
# algebraic functions of the sigmas, so they are computed once after the run from
# t_sigma_{x,y,z}.dat rather than inside the evolution loop.

import os

import numpy as np


def rho_components(sigma_x, sigma_y, sigma_z):
    """
    Single-site density-matrix components from single-site Pauli expectations.

    rho = 1/2 (I + <sx> X + <sy> Y + <sz> Z), so per site:
        rho_ee   = (sigma_z + 1) / 2
        rho_mumu = (1 - sigma_z) / 2
        rho_emu  = 0.5 * sqrt(sigma_x^2 + sigma_y^2)   ( = |rho[0,1]| )

    Works elementwise on arrays. Shared by the sampled and exact-state sigma
    paths (both via rho_from_sigmas) so the formula lives in one place.
    """
    rho_ee = (sigma_z + 1.0) / 2.0
    rho_mumu = (1.0 - sigma_z) / 2.0
    rho_emu = 0.5 * np.sqrt(sigma_x**2 + sigma_y**2)
    return rho_ee, rho_mumu, rho_emu


def rho_from_sigmas(datadir, sigma_suffix="", rho_suffix="from_counts"):
    """
    Compute single-site density-matrix components from a set of sigma files.

    Reads t_sigma_x{sigma_suffix}.dat, t_sigma_y{sigma_suffix}.dat,
    t_sigma_z{sigma_suffix}.dat from datadir. Each file has one row per timestep:
    a leading time column followed by the per-site <sigma_b> in original particle
    order. Applies rho_components per site and writes
    t_rho_ee_{rho_suffix}.dat, t_rho_mumu_{rho_suffix}.dat, and
    t_rho_emu_{rho_suffix}.dat (time column + per-site values).

    The same transform serves both paths:
      - sampled sigmas:  rho_from_sigmas(datadir)                       -> *_from_counts
      - exact  sigmas:   rho_from_sigmas(datadir, "_direct", "direct")  -> *_direct

    Inputs:
    - datadir: directory holding the sigma files; the rho files are written back
      into it.
    - sigma_suffix: suffix on the input t_sigma_* file names ("" or "_direct").
    - rho_suffix: suffix on the output t_rho_* file names.

    Output:
    - (times, rho_ee, rho_mumu, rho_emu): the time array and the three per-site
      component arrays (shape [n_times, n_sites]).
    """
    def load(basis):
        return np.atleast_2d(
            np.loadtxt(os.path.join(datadir, f"t_sigma_{basis}{sigma_suffix}.dat"))
        )

    sx_data = load("x")
    sy_data = load("y")
    sz_data = load("z")

    if not (sx_data.shape == sy_data.shape == sz_data.shape):
        raise ValueError(
            f"t_sigma_{{x,y,z}}{sigma_suffix}.dat have mismatched shapes "
            f"({sx_data.shape}, {sy_data.shape}, {sz_data.shape}); "
            "they must share the same times and sites."
        )

    times = sz_data[:, 0]
    sx = sx_data[:, 1:]
    sy = sy_data[:, 1:]
    sz = sz_data[:, 1:]

    rho_ee, rho_mumu, rho_emu = rho_components(sx, sy, sz)

    for fname, rho in (
        (f"t_rho_ee_{rho_suffix}.dat", rho_ee),
        (f"t_rho_mumu_{rho_suffix}.dat", rho_mumu),
        (f"t_rho_emu_{rho_suffix}.dat", rho_emu),
    ):
        np.savetxt(os.path.join(datadir, fname), np.column_stack([times, rho]), fmt="%.16e")

    return times, rho_ee, rho_mumu, rho_emu
