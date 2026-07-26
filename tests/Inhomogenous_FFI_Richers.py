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

from constants import hbar, c, MeV
from activate_backend import activate_backends, build_backend, backend_supports_direct_state
from time_evolution import run_time_evolution
from observables import SigmaObservable, DirectSigmaObservable
from rho_from_counts import rho_from_sigmas


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq_simulator(ideal simulator non-native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm
####FYI : Fake manila works only for qubits: 4 or less since FakeManila is a 5-qubit device model.
trotter_order = "second" #first, second
ibm_service, ionq_provider = activate_backends()


def initialize_parameters(ibm_service, ionq_provider):
    params = {}

    N_sites_eachflavor = 5  # evenly spaced sites per (electron) flavor
    params["N_sites"] = 2 * N_sites_eachflavor
    N_sites = params["N_sites"]

    # --- masses / mixing (both zero here: pure FFI, no vacuum term) ---
    m1 = 0.0
    m2 = 0
    params["delta_m_squared"] = (m2**2 - m1**2)  # = 0
    params["theta_nu"] = 1.74532925E-8           # mixing angle (~1e-6 deg) in radians

    # --- run controls ---
    params["shots"] = 1024
    params["trotter_steps"] = 10  # Trotter steps per time step
    params["trotter_order"] = trotter_order
    params["optimization_level"] = 1
    params["df"] = 1
    params["datadir"] = os.path.join(os.getcwd(), "datafiles")
    params["observables"] = [
        SigmaObservable("X"),
        SigmaObservable("Y"),
        SigmaObservable("Z"),
        DirectSigmaObservable("X"),
        DirectSigmaObservable("Y"),
        DirectSigmaObservable("Z"),
    ]
    params["backend_name"] = backend_name

    # --- time grid ---
    params["tau"] = 5E-12
    params["ttotal"] = 1.75E-10
    params["times"] = np.arange(0, params["ttotal"], params["tau"])
    # growth-rate fit window (used in post-processing)
    params["t1"] = 6e-11
    params["t2"] = 1.5e-10

    # --- geometry / box ---
    L = 1  # cm, domain size
    params["L"] = L
    params["dx"] = L / N_sites_eachflavor  # site box length
    params["dp"] = params["dx"]            # shape-function width
    params["shape_name"] = "triangular"
    params["geometric_name"] = "physical"
    params["periodic"] = True
    params["advection"] = True

    # --- perturbation (this test seeds one, unlike the homogeneous case) ---
    theta_nu = params["theta_nu"]
    theta_pert = theta_nu
    B_pert = np.array([-np.sin(2 * theta_pert), 0, np.cos(2 * theta_pert)])
    params["B_pert"] = B_pert / np.linalg.norm(B_pert)
    params["alpha"] = 1e-2
    print("B_pert =", params["B_pert"])

    # --- backend ---
    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=1,
        matrix_product_state_truncation_threshold=1e-100,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    (
        params["backend"],
        params["euler_basis"],
        params["basis_gates"],
        params["two_qubit_gate"],
    ) = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
    )

    # --- number densities -> particle counts per site ---
    n_νₑ = 4.891290848285061e+32  # cm^-3, electron-neutrino number density
    n_νₑ̄ = n_νₑ                   # cm^-3, electron-antineutrino number density
    V = params["dx"]**3           # volume of a site box
    N_1 = np.full(N_sites // 2, n_νₑ * V)
    N_2 = np.full(N_sites // 2, n_νₑ̄ * V)
    params["N"] = np.concatenate((N_1, N_2))
    print("N =", params["N"])

    # --- B field (same for all particles) ---
    B = np.array([-np.sin(2 * theta_nu), 0, np.cos(2 * theta_nu)])
    params["B"] = B / np.linalg.norm(B)

    # --- positions: neutrinos and antineutrinos share the same grid ---
    def generate_x_array(N_sites_eachflavor, L):
        one_flavor = [(i - 0.5) * L / N_sites_eachflavor for i in range(1, N_sites_eachflavor + 1)]
        return one_flavor + one_flavor  # mu then anti-mu, same positions

    params["x"] = generate_x_array(N_sites_eachflavor, L)
    params["y"] = generate_x_array(N_sites_eachflavor, L)
    params["z"] = generate_x_array(N_sites_eachflavor, L)

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


def chi_squareg_test(par, tt, shots):
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


def simulate(params):
    datadir = params["datadir"]

    # Driver: sorts by position, builds the (perturbed) circuit, and each step
    # samples sigma_{x,y,z} (-> t_sigma_*.dat) and records the exact single-site
    # sigmas (-> t_sigma_*_direct.dat), plus the position/momentum files.
    run_time_evolution(params)

    # Sampled rho is a post-processing transform of the sampled sigma files.
    rho_from_sigmas(datadir)                        # sampled -> t_rho_*_from_counts.dat

    # The exact sigma files (and hence direct rho) exist only on an exact backend.
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

    # analytic growth rate: vacuum term + advection term c*k (k = 2 pi / L)
    k = 2 * np.pi / params["L"]
    analytic_growth_rate = abs(params["delta_m_squared"]) / (2 * hbar * params["Enue"]) + c * k

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

    # NOTE: this test prints growth rates but has no pass/fail assert (it never
    # had one). It cannot fail on any backend; add an aer-gated assert on
    # Im_omega_direct vs analytic if a real check is wanted.
    return results


def plot_results(results, params, analytic_offset=100.0):
    N_sites = params["N_sites"]
    times = np.asarray(results["times"], dtype=float)
    backend = params["backend"]
    backend_name = params["backend_name"]
    analytic_growth_rate = results["analytic_growth_rate"]
    fit_t1, fit_t2 = params["t1"], params["t2"]

    plotdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plotdir, exist_ok=True)

    def safe_semilogy_y(y):
        y = np.abs(np.asarray(y, dtype=float))
        y[~np.isfinite(y)] = np.nan
        y[y <= 0] = np.nan
        return y

    def get_mps_bond_dimension_from_backend(backend):
        if backend is None:
            return None
        try:
            return backend.options.matrix_product_state_max_bond_dimension
        except Exception:
            return None

    def plot_all_sites(data, ylabel, title, filename, semilogy=False):
        plt.figure()
        for site in range(N_sites):
            y = data[:, site]
            if semilogy:
                plt.semilogy(times, safe_semilogy_y(y), label=f"site {site+1}")
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
                   "Inhomo_sigma_x_vs_time.pdf")
    plot_all_sites(results["sigma_y"], r"$\sigma_y$",
                   fr"$\sigma_y$ vs Time for all sites ({N_sites} sites)",
                   "Inhomo_sigma_y_vs_time.pdf")
    plot_all_sites(results["sigma_z"], r"$\sigma_z$",
                   fr"$\sigma_z$ vs Time for all sites ({N_sites} sites)",
                   "Inhomo_sigma_z_vs_time.pdf")

    # ========================= Rho plots from counts ===========================
    plot_all_sites(results["rho_ee"], r"$\rho_{ee}$",
                   fr"$\rho_{{ee}}$ from counts vs Time for all sites ({N_sites} sites)",
                   "Inhomo_rho_ee_counts_vs_time.pdf", semilogy=True)
    plot_all_sites(results["rho_mumu"], r"$\rho_{\mu\mu}$",
                   fr"$\rho_{{\mu\mu}}$ from counts vs Time for all sites ({N_sites} sites)",
                   "Inhomo_rho_mumu_counts_vs_time.pdf", semilogy=True)
    plot_all_sites(results["rho_emu"], r"$|\rho_{e\mu}|$",
                   fr"$|\rho_{{e\mu}}|$ from counts vs Time for all sites ({N_sites} sites)",
                   "Inhomo_rho_emu_counts_vs_time.pdf", semilogy=True)

    # ========================= Direct rho plots (exact backend only) ===========
    if "rho_emu_direct" not in results:
        # position / momentum plots still make sense; jump to them
        _plot_positions(results, params, plotdir, times)
        return

    plot_all_sites(results["rho_ee_direct"], r"$\rho_{ee}^{direct}$",
                   fr"Direct $\rho_{{ee}}$ vs Time for all sites ({N_sites} sites)",
                   "Inhomo_rho_ee_direct_vs_time.pdf", semilogy=True)
    plot_all_sites(results["rho_mumu_direct"], r"$\rho_{\mu\mu}^{direct}$",
                   fr"Direct $\rho_{{\mu\mu}}$ vs Time for all sites ({N_sites} sites)",
                   "Inhomo_rho_mumu_direct_vs_time.pdf", semilogy=True)

    # ============ Direct rho_emu plot + shifted analytic line ==================
    rho_emu_direct = np.asarray(results["rho_emu_direct"], dtype=float)
    plt.figure()
    for site in range(N_sites):
        plt.semilogy(times, safe_semilogy_y(rho_emu_direct[:, site]), label=f"site {site+1}")

    bond_dim = get_mps_bond_dimension_from_backend(backend)
    should_plot_analytic_line = (
        backend_name == "aer" and bond_dim == 1 and analytic_growth_rate is not None
    )

    if should_plot_analytic_line:
        rho_target = safe_semilogy_y(rho_emu_direct[:, 0])
        mask_fit = (times >= fit_t1) & (times <= fit_t2)
        x_fit = times[mask_fit]
        y_fit = rho_target[mask_fit]
        mask_valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (y_fit > 0)

        if np.count_nonzero(mask_valid) >= 2:
            x_fit = x_fit[mask_valid]
            t_ref = x_fit[-1]
            ymax_data = np.nanmax(rho_emu_direct)
            y_top = analytic_offset * ymax_data
            y_analytic = y_top * np.exp(analytic_growth_rate * (x_fit - t_ref))

            plt.semilogy(x_fit, y_analytic, "--", color="gray", linewidth=2,
                         label=fr"analytic growth: {analytic_growth_rate:.2e} s$^{{-1}}$")
            mid_idx = len(x_fit) // 2
            plt.text(x_fit[mid_idx], y_analytic[mid_idx] * 6,
                     fr"${analytic_growth_rate:.2e}\ \mathrm{{s}}^{{-1}}$",
                     color="gray", fontsize=10, rotation=45, ha="center", va="center")
            print(f"Plotted shifted analytic line from t={x_fit[0]:.3e} to t={x_fit[-1]:.3e} "
                  f"with offset={analytic_offset}")
        else:
            print("Skipping analytic line: not enough positive finite rho_emu_direct points "
                  f"between {fit_t1:.3e} and {fit_t2:.3e}.")

    plt.xlabel("Time")
    plt.ylabel(r"$|\rho_{e\mu}^{direct}|$")
    plt.title(fr"Direct $|\rho_{{e\mu}}|$ vs Time for all sites ({N_sites} sites)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Inhomo_rho_emu_direct_vs_time.pdf"), bbox_inches="tight")
    plt.show()

    _plot_positions(results, params, plotdir, times)


def _plot_positions(results, params, plotdir, times):
    N_sites = params["N_sites"]

    # ========================= Position evolution ==============================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.xlabel("Position (x)")
    plt.ylabel("Time")
    for site in range(N_sites):
        plt.plot(results["x_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Inhomo_Particles_position_evolution.pdf"), bbox_inches="tight")
    plt.show()

    # ========================= Momentum evolution ==============================
    plt.figure()
    plt.title(f"Momentum Evolution for {N_sites} sites")
    plt.xlabel(r"Momentum in x direction ($p_x$)")
    plt.ylabel("Time")
    for site in range(N_sites):
        plt.plot(results["px_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Inhomo_Particles_momentum_evolution.pdf"), bbox_inches="tight")
    plt.show()


params = initialize_parameters(ibm_service, ionq_provider)
results = simulate(params)
# chi_square_value, *_ = chi_squareg_test(params["N_sites"], params["ttotal"], params["shots"])
plot_results(results, params)
