#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Determine the correct path for the 'src' directory
if "__file__" in globals():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
else:
    src_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

from time_evolution import run_time_evolution
from observables import (
    PositionObservable,
    MomentumObservable,
    SigmaObservable,
    DirectSigmaObservable,
)
from momentum import vacuum_frequency
from constants import hbar, eV, MeV, G_F
from activate_backend import activate_backends, build_backend


backend_name = "aer_MB" #manila,guadalupe, aer_MF, aer_MB, ibm, ionq
trotter_order= "first" #first, second
save_plots_flag = False

ibm_service, ionq_provider = activate_backends()


def initialize_parameters(ibm_service, ionq_provider):
    params = {}
    params["N_sites"] = 4
    N_sites = params["N_sites"]  # alias, used below to build the per-site arrays

    # --- non-derived scalar inputs ---
    params["L"] = 1.0
    params["ttotal"] = 1.94e-4
    params["tolerance"] = 5e-2
    params["shots"] = 1000
    params["trotter_steps"] = 1
    params["optimization_level"] = 1
    params["df"] = 0
    params["tau"] = 3.88e-6  # (ttotal/50 from before)
    params["dx"] = 1e-3
    params["dp"] = params["L"]
    params["theta_nu"] = np.pi / 4
    params["shape_name"] = "none"
    params["geometric_name"] = "physical"
    params["periodic"] = False
    params["advection"] = False  # True/False
    params["backend_name"] = backend_name
    params["trotter_order"] = trotter_order
    params["datadir"] = os.path.join(os.getcwd(), "datafiles")
    params["observables"] = [
        PositionObservable(),
        MomentumObservable("x"),
        MomentumObservable("y"),
        MomentumObservable("z"),
        SigmaObservable("X"),
        SigmaObservable("Y"),
        SigmaObservable("Z"),
        DirectSigmaObservable("X"),
        DirectSigmaObservable("Y"),
        DirectSigmaObservable("Z"),
    ]

    # Optional perturbation parameters
    params["B_pert"] = None
    params["alpha"] = None

    # --- backend (derived) ---
    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=1,
        matrix_product_state_truncation_threshold=1e-100,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    params["backend"] = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
    )

    # --- derived arrays / values ---
    params["times"] = np.arange(0.0, params["ttotal"] + 0.5 * params["tau"], params["tau"])

    # Julia:
    # m1 = 0 * eV
    # m2 = 0.008596511 * eV
    m1 = 0.0 * eV
    m2 = 0.008596511 * eV
    params["delta_m_squared"] = abs(m2**2 - m1**2)

    # All sites are neutrinos
    params["energy_sign"] = np.ones(N_sites)

    params["p"] = np.ones((N_sites, 3), dtype=float) * MeV

    mu = np.zeros(N_sites, dtype=float)
    params["N"] = mu * np.full(N_sites, (params["dx"] ** 3) / (np.sqrt(2) * G_F), dtype=float)

    x0 = np.random.rand()
    y0 = np.random.rand()
    params["x"] = np.full(N_sites, x0, dtype=float)
    params["y"] = np.full(N_sites, y0, dtype=float)
    params["z"] = np.zeros(N_sites, dtype=float)

    # omega is derived inside run_time_evolution via vacuum_frequency; not stored
    # here (the post-processing recomputes it for the analytic comparison).

    # Initial product state: first half Dn, second half Up
    # Use strings because later we join them into a bitstring
    params["bit_list"] = ["1" if i < N_sites // 2 else "0" for i in range(N_sites)]

    return params


def simulate(params):
    datadir = params["datadir"]

    # Run the shared driver. It sorts by position, builds the base circuit, and
    # each step calls the recorders in params["observables"] -- writing
    # t_sigma_{x,y,z}{,_direct}.dat alongside the position/momentum files and
    # run_parameters.json, all into datadir.
    run_time_evolution(params)

    # ----------------------------------------------------------------------
    # Post-processing: site-1 <Sz> vs the analytic vacuum-oscillation curve.
    # These are functions of the sampled sigmas + the (constant) frequency, so
    # they are computed here from the files rather than inside the loop.
    # ----------------------------------------------------------------------
    times = np.asarray(params["times"], dtype=float)
    tolerance = params["tolerance"]

    def load_sigma(basis):
        path = os.path.join(datadir, f"t_sigma_{basis}.dat")
        return np.atleast_2d(np.loadtxt(path))[:, 1:]  # drop the leading time column

    sigma_x_all_times = load_sigma("x")
    sigma_y_all_times = load_sigma("y")
    sigma_z_all_times = load_sigma("z")

    # site 1 = first site in original particle order = column 0 of the sigma files
    sigma_z_site1_values = sigma_z_all_times[:, 0]
    Sz_site1_values = 0.5 * sigma_z_site1_values

    # Analytic vacuum oscillation (Sakurai): <Sz>(t) = -1/2 cos(omega t / hbar).
    # All sites share the same |p| and energy_sign here, so omega is uniform and
    # constant in time; take site 0.
    omega = vacuum_frequency(params["delta_m_squared"], params["p"], params["energy_sign"])
    expected_Sz_site1 = -0.5 * np.cos(omega[0] / hbar * times)
    expected_sigma_z_site1 = 2.0 * expected_Sz_site1

    def save_col(name, col):
        np.savetxt(os.path.join(datadir, name), np.column_stack([times, col]), fmt="%.16e")

    save_col("t_sigma_z_site1.dat", sigma_z_site1_values)
    save_col("t_Sz_site1.dat", Sz_site1_values)
    save_col("t_expected_sigma_z_site1.dat", expected_sigma_z_site1)
    save_col("t_expected_Sz_site1.dat", expected_Sz_site1)

    max_abs_err = np.max(np.abs(Sz_site1_values - expected_Sz_site1))
    print("max |<Sz>_site1 - analytic| =", max_abs_err)

    if params["backend_name"] == "aer":
        assert np.all(np.abs(Sz_site1_values - expected_Sz_site1) < tolerance), (
            "Assertion failed: site-1 <Sz> values differ from the analytic "
            f"curve by more than tolerance = {tolerance}"
        )

    return {
        "sigma_x_all_times": sigma_x_all_times,
        "sigma_y_all_times": sigma_y_all_times,
        "sigma_z_all_times": sigma_z_all_times,
        "sigma_z_site1_values": sigma_z_site1_values,
        "Sz_site1_values": Sz_site1_values,
        "expected_sigma_z_site1": expected_sigma_z_site1,
        "expected_Sz_site1": expected_Sz_site1,
        "times": times,
    }


def plot_results(results, params):
    times = results["times"]
    Sz_site1_values = results["Sz_site1_values"]
    expected_Sz_site1 = results["expected_Sz_site1"]

    plt.figure(figsize=(7, 6))
    plt.plot(times, Sz_site1_values, label="My_sz_site1 from counts")
    plt.plot(times, expected_Sz_site1, ":", label="Expected_sz from Sakurai_site1")
    plt.xlabel("t")
    plt.ylabel("<Sz>")
    plt.title("Running main_vac_osc script")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    plt.savefig("main_vac_osc_test.pdf", dpi=300, bbox_inches="tight")

    plt.show()


params = initialize_parameters(ibm_service, ionq_provider)
results = simulate(params)
plot_results(results, params)
