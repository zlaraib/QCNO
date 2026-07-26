#!/usr/bin/env python
# coding: utf-8

import os

# Limit the number of cores (must be set before numpy / Aer import)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
print("MKL_NUM_THREADS:", os.environ["MKL_NUM_THREADS"])
print("JULIA_NUM_THREADS:", os.environ["JULIA_NUM_THREADS"])

import numpy as np
import matplotlib.pyplot as plt
import sys

# Determine the correct path for the 'src' directory
if '__file__' in globals():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
else:
    src_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))

sys.path.append(src_dir)

from constants import hbar, eV, MeV
from activate_backend import activate_backends, build_backend, backend_supports_direct_state
from time_evolution import run_time_evolution
from observables import (
    PositionObservable,
    MomentumObservable,
    SigmaObservable,
    DirectSigmaObservable,
)
from rho_from_counts import rho_from_sigmas


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq_simulator(ideal simulator non-native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm
####FYI : Fake manila works only for qubits: 4 or less since FakeManila is a 5-qubit device model.
trotter_order = "second" #first, second
ibm_service, ionq_provider = activate_backends()


def initialize_parameters(ibm_service, ionq_provider):
    params = {}

    # total sites/particles/qubits, half electron neutrinos and half antineutrinos
    N_sites_eachflavor = 1
    params["N_sites"] = 2 * N_sites_eachflavor
    N_sites = params["N_sites"]

    # --- masses / mixing ---
    m1 = 0 * eV                 # 1st mass eigenstate (Richers 2021)
    m2 = 0.008596511 * eV       # 2nd mass eigenstate (Richers 2021)
    params["delta_m_squared"] = (m2**2 - m1**2)  # erg^2
    params["theta_nu"] = 1.74532925E-8  # mixing angle (~1e-6 deg) in radians

    # --- run controls ---
    params["shots"] = 1042
    params["trotter_steps"] = 10  # Trotter steps per time step
    params["trotter_order"] = trotter_order
    params["optimization_level"] = 1
    params["df"] = 1              # degrees of freedom for chi-square test
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
    params["tolerance"] = 1e-1
    params["backend_name"] = backend_name

    # --- time grid ---
    params["tau"] = 1.666e-4
    params["ttotal"] = 1.666e-2
    params["times"] = np.arange(0, params["ttotal"], params["tau"])
    # growth-rate fit window (used in post-processing)
    params["t1"] = 0.013328000000000001
    params["t2"] = 0.014827400000000001

    # --- geometry / box ---
    L = 1e7                      # cm, domain size (big box length)
    params["L"] = L
    params["dx"] = L             # box length at a site
    params["dp"] = L             # shape-function width (unused here, kept for the evolve args)
    params["shape_name"] = "none"
    params["geometric_name"] = "physical"
    params["periodic"] = True
    params["advection"] = False
    params["B_pert"] = None
    params["alpha"] = 0

    # --- backend ---
    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=1,
        matrix_product_state_truncation_threshold=1e-100,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    params["backend"] = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
        ibm_backend="least_busy",
    )

    # --- number densities -> particle counts per site ---
    n_νₑ = 2.92e24               # cm^-3, electron-neutrino number density
    n_νₑ̄ = n_νₑ                  # cm^-3, electron-antineutrino number density
    V = L**3                     # volume of the big box
    N_1 = np.full(N_sites // 2, n_νₑ * V)
    N_2 = np.full(N_sites // 2, n_νₑ̄ * V)
    params["N"] = np.concatenate((N_1, N_2))

    # --- B field (same for all particles) ---
    theta_nu = params["theta_nu"]
    B = np.array([-np.sin(2 * theta_nu), 0, np.cos(2 * theta_nu)])
    params["B"] = B / np.linalg.norm(B)

    # --- positions (grid, first particle at L/(2 N_sites)) ---
    def generate_x_array(N_sites, L):
        return [(i - 0.5) * L / N_sites for i in range(1, N_sites + 1)]

    params["x"] = generate_x_array(N_sites, L)
    params["y"] = generate_x_array(N_sites, L)
    params["z"] = generate_x_array(N_sites, L)

    # --- momenta: neutrinos and antineutrinos in opposing beams ---
    Eνₑ = 50 * MeV               # energy of all neutrinos
    Eνₑ̄ = -1 * Eνₑ              # antineutrinos move opposite -> negative sign
    params["Enue"] = Eνₑ         # kept for the analytic growth rate
    half = N_sites // 2
    px = np.concatenate([np.full(half, Eνₑ), np.full(half, Eνₑ̄)])
    py = np.zeros(N_sites)
    pz = np.zeros(N_sites)
    params["p"] = np.column_stack((px, py, pz))

    # first half antineutrinos (-1), second half neutrinos (+1)
    params["energy_sign"] = np.array([-1 if i < N_sites // 2 else 1 for i in range(N_sites)])
    params["bit_list"] = ['0' if s == -1 else '1' for s in params["energy_sign"]]

    return params


def simulate(params):
    datadir = params["datadir"]

    # Driver: sorts by position, builds the circuit, and each step samples
    # sigma_{x,y,z} (-> t_sigma_*.dat) and records the exact single-site sigmas
    # (-> t_sigma_*_direct.dat), plus the position/momentum files.
    run_time_evolution(params)

    # Sampled rho is a post-processing transform of the sampled sigma files.
    rho_from_sigmas(datadir)                        # sampled -> t_rho_*_from_counts.dat

    # The exact sigma files (and hence the direct rho) exist only when the backend
    # has an exact state.
    direct_available = backend_supports_direct_state(params["backend"])
    if direct_available:
        rho_from_sigmas(datadir, "_direct", "direct")   # exact -> t_rho_*_direct.dat

    def load_cols(name):
        return np.atleast_2d(np.loadtxt(os.path.join(datadir, name)))

    # ---------------------------------------------------------------
    # Fast-flavor growth rate from |rho_emu| at site 0 (counts + direct)
    # ---------------------------------------------------------------
    rho_emu_counts = load_cols("t_rho_emu_from_counts.dat")
    times = rho_emu_counts[:, 0]

    t1, t2 = params["t1"], params["t2"]
    del_t = abs(t2 - t1)

    # nearest sample indices to the fit window (same times for counts and direct)
    i1 = int(np.argmin(np.abs(times - t1)))
    i2 = int(np.argmin(np.abs(times - t2)))
    print(f"fit window: times[{i1}]={times[i1]:.6e}, times[{i2}]={times[i2]:.6e}")

    def growth_rate(rho_emu_site0, eps=1e-300):
        """Im(omega) from |rho_emu| at site 0 over the fit window, floored at eps."""
        r1 = max(rho_emu_site0[i1], eps)
        r2 = max(rho_emu_site0[i2], eps)
        return (1.0 / del_t) * abs(np.log(r2 / r1))

    Im_omega = growth_rate(rho_emu_counts[:, 1])

    analytic_growth_rate = abs(params["delta_m_squared"]) / (2 * hbar * params["Enue"])

    print(f"Growth rate from counts-based rho_emu: Im(omega) = {Im_omega}")
    print(f"Analytic growth rate: Im(omega) = {analytic_growth_rate}")

    results = {
        "times": times,
        "sigma_x": load_cols("t_sigma_x.dat")[:, 1:],
        "sigma_y": load_cols("t_sigma_y.dat")[:, 1:],
        "sigma_z": load_cols("t_sigma_z.dat")[:, 1:],
        "rho_ee": load_cols("t_rho_ee_from_counts.dat")[:, 1:],
        "rho_mumu": load_cols("t_rho_mumu_from_counts.dat")[:, 1:],
        "rho_emu": load_cols("t_rho_emu_from_counts.dat")[:, 1:],
        "x_values": load_cols("t_xsiteval.dat")[:, 1:],
        "px_values": load_cols("t_pxsiteval.dat")[:, 1:],
        "Im_omega": Im_omega,
        "analytic_growth_rate": analytic_growth_rate,
    }

    if direct_available:
        rho_emu_direct = load_cols("t_rho_emu_direct.dat")
        Im_omega_direct = growth_rate(rho_emu_direct[:, 1])
        print(f"Growth rate from direct rho_emu: Im(omega) = {Im_omega_direct}")

        results["Im_omega_direct"] = Im_omega_direct
        results["rho_ee_direct"] = load_cols("t_rho_ee_direct.dat")[:, 1:]
        results["rho_mumu_direct"] = load_cols("t_rho_mumu_direct.dat")[:, 1:]
        results["rho_emu_direct"] = load_cols("t_rho_emu_direct.dat")[:, 1:]

        # The direct growth rate is the pass/fail gate, and is only meaningful on
        # the exact simulator (aer).
        if params["backend_name"] == "aer":
            assert abs((Im_omega_direct - analytic_growth_rate) / analytic_growth_rate) < params["tolerance"], \
                "Direct growth rate deviates from analytic result beyond tolerance"

    return results


def chi_square_test(par, tt, shots):
    # File paths
    file_path1 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/shots_{shots}/t_<s_z>_<s_y>_<s_x>.dat"
    # file_path2 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/t_<Sz>_<Sy>_<Sx>_CCold.dat"
    file_path2 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/t_<Sz>_<Sy>_<Sx>_MF.dat"
    file_path3 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/shots_{shots}/t_sigma_xbar_i.dat"
    file_path4 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/shots_{shots}/t_sigma_ybar_i.dat"
    file_path5 = f"../misc/datafiles/FFI/par_{par}/tt_{tt}/shots_{shots}/t_sigma_zbar_i.dat"

    # Read and load the data from the files
    s_data_QC = np.loadtxt(file_path1)  # observed data (sigma_x_QC)
    S_data_CC = np.loadtxt(file_path2)  # expected data (sigma_x_CC)
    sigma_xi_data_QC = np.loadtxt(file_path3)  # standard deviation (sigma_xbar_i)
    sigma_yi_data_QC = np.loadtxt(file_path4)  # standard deviation (sigma_ybar_i)
    sigma_zi_data_QC = np.loadtxt(file_path5)  # standard deviation (sigma_zbar_i)

    # Extract the arrays
    times_CC = S_data_CC[:, 0]
    times_QC = s_data_QC[:, 0]
    times_sigma_xi_QC = sigma_xi_data_QC[:, 0]

    sx_array_QC = s_data_QC[:, 3]* 1/2  # observed values (y_i) # divided by 2 to accomodate the factor of 2 coming in Sx classical simulation
    Sx_array_CC = S_data_CC[:, 3]  # expected values (f(x_i))
    sy_array_QC = s_data_QC[:, 2]* 1/2  # observed values (y_i) # divided by 2 to accomodate the factor of 2 coming in Sy classical simulation
    Sy_array_CC = S_data_CC[:, 2]  # expected values (f(x_i))
    sz_array_QC = s_data_QC[:, 1]* 1/2  # observed values (y_i) # divided by 2 to accomodate the factor of 2 coming in Sz classical simulation
    Sz_array_CC = S_data_CC[:, 1]  # expected values (f(x_i))

    sigma_xi_array_QC = sigma_xi_data_QC[:, 1]  # standard deviations (sigma_xi)
    sigma_yi_array_QC = sigma_yi_data_QC[:, 1]  # standard deviations (sigma_yi)
    sigma_zi_array_QC = sigma_zi_data_QC[:, 1]  # standard deviations (sigma_zi)

    print(f"Sx_array_CC: {Sx_array_CC}")
    print(f"sx_array_QC: {sx_array_QC}")
    print(f"sigma_xi_array_QC: {sigma_xi_array_QC}")

    print(f"Sy_array_CC: {Sy_array_CC}")
    print(f"sy_array_QC: {sy_array_QC}")
    print(f"sigma_yi_array_QC: {sigma_yi_array_QC}")

    print(f"Sz_array_CC: {Sz_array_CC}")
    print(f"sz_array_QC: {sz_array_QC}")
    print(f"sigma_zi_array_QC: {sigma_zi_array_QC}")

    # Ensure the arrays are of the same length
    if len(sx_array_QC) != len(Sx_array_CC) or len(sx_array_QC) != len(sigma_xi_array_QC):
        raise ValueError("Arrays must be of the same length for chi-square calculation")

    # Compute chi-square value
    chi_square_value = np.mean(((Sx_array_CC - sx_array_QC ) / sigma_xi_array_QC) ** 2)
    print(f"Chi-square value: {chi_square_value}")

    # Ensure the arrays are of the same length
    if len(sy_array_QC) != len(Sy_array_CC) or len(sy_array_QC) != len(sigma_yi_array_QC):
        raise ValueError("Arrays must be of the same length for chi-square calculation")

    # Compute chi-square value
    chi_square_value = np.mean(((Sy_array_CC - sy_array_QC ) / sigma_yi_array_QC) ** 2)
    print(f"Chi-square value: {chi_square_value}")

    # Ensure the arrays are of the same length
    if len(sz_array_QC) != len(Sz_array_CC) or len(sz_array_QC) != len(sigma_zi_array_QC):
        raise ValueError("Arrays must be of the same length for chi-square calculation")

    # Compute chi-square value
    chi_square_value = np.mean(((Sz_array_CC - sz_array_QC ) / sigma_zi_array_QC) ** 2)
    print(f"Chi-square value: {chi_square_value}")

    # Assert that chi-square value is less than or equal to 1 sigma (std of mean)
    # assert chi_square_value <= 1, f"Chi-square value is too high: {chi_square_value}. It should be <= 1 sigma."

    return chi_square_value, Sx_array_CC, sx_array_QC, sigma_xi_array_QC, Sy_array_CC, sy_array_QC, sigma_yi_array_QC, Sz_array_CC, sz_array_QC, sigma_zi_array_QC


def plot_results(results, params):
    N_sites = params["N_sites"]
    times = results["times"]

    plotdir = os.path.join(os.getcwd(), "plots")
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    def plot_all_sites(data, ylabel, title, filename, semilogy=False):
        plt.figure()
        for site in range(N_sites):
            y = data[:, site]
            if semilogy:
                plt.semilogy(times, y, label=f"site {site+1}")
            else:
                plt.plot(times, y, label=f"site {site+1}")
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plotdir, filename), bbox_inches="tight")
        plt.show()

    # ========================= Sigma plots from counts =========================
    plot_all_sites(results["sigma_x"], r"$\sigma_x$",
                   fr"$\sigma_x$ vs Time for all sites ({N_sites} sites)",
                   "Homo_sigma_x_vs_time.pdf")
    plot_all_sites(results["sigma_y"], r"$\sigma_y$",
                   fr"$\sigma_y$ vs Time for all sites ({N_sites} sites)",
                   "Homo_sigma_y_vs_time.pdf")
    plot_all_sites(results["sigma_z"], r"$\sigma_z$",
                   fr"$\sigma_z$ vs Time for all sites ({N_sites} sites)",
                   "Homo_sigma_z_vs_time.pdf")

    # ========================= Rho plots from counts ===========================
    plot_all_sites(results["rho_ee"], r"$\rho_{ee}$",
                   fr"$\rho_{{ee}}$ from counts vs Time for all sites ({N_sites} sites)",
                   "Homo_rho_ee_counts_vs_time.pdf", semilogy=True)
    plot_all_sites(results["rho_mumu"], r"$\rho_{\mu\mu}$",
                   fr"$\rho_{{\mu\mu}}$ from counts vs Time for all sites ({N_sites} sites)",
                   "Homo_rho_mumu_counts_vs_time.pdf", semilogy=True)
    plot_all_sites(results["rho_emu"], r"$|\rho_{e\mu}|$",
                   fr"$|\rho_{{e\mu}}|$ from counts vs Time for all sites ({N_sites} sites)",
                   "Homo_rho_emu_counts_vs_time.pdf", semilogy=True)

    # ========================= Direct rho plots ================================
    # Only present when the run used an exact simulator.
    if "rho_ee_direct" in results:
        plot_all_sites(results["rho_ee_direct"], r"$\rho_{ee}^{direct}$",
                       fr"Direct $\rho_{{ee}}$ vs Time for all sites ({N_sites} sites)",
                       "Homo_rho_ee_direct_vs_time.pdf", semilogy=True)
        plot_all_sites(results["rho_mumu_direct"], r"$\rho_{\mu\mu}^{direct}$",
                       fr"Direct $\rho_{{\mu\mu}}$ vs Time for all sites ({N_sites} sites)",
                       "Homo_rho_mumu_direct_vs_time.pdf", semilogy=True)
        plot_all_sites(results["rho_emu_direct"], r"$|\rho_{e\mu}^{direct}|$",
                       fr"Direct $|\rho_{{e\mu}}|$ vs Time for all sites ({N_sites} sites)",
                       "Homo_rho_emu_direct_vs_time.pdf", semilogy=True)

    # ========================= Position evolution ==============================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.xlabel("Time")
    plt.ylabel("Position (x)")
    for site in range(N_sites):
        plt.plot(results["x_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Homo_Particles_position_evolution.pdf"), bbox_inches="tight")
    plt.show()

    # ========================= Momentum evolution ==============================
    plt.figure()
    plt.title(f"Momentum Evolution for {N_sites} sites")
    plt.xlabel("Time")
    plt.ylabel(r"Momentum in x direction ($p_x$)")
    for site in range(N_sites):
        plt.plot(results["px_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Homo_Particles_momentum_evolution.pdf"), bbox_inches="tight")
    plt.show()


params = initialize_parameters(ibm_service, ionq_provider)
results = simulate(params)
# chi_square_value, *_ = chi_square_test(params["N_sites"], params["ttotal"], params["shots"])
plot_results(results, params)
