#!/usr/bin/env python
# coding: utf-8

import os

# Limit the number of cores (must be set before numpy / Aer import)
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["JULIA_NUM_THREADS"] = "10"

print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
print("MKL_NUM_THREADS:", os.environ["MKL_NUM_THREADS"])
print("JULIA_NUM_THREADS:", os.environ["JULIA_NUM_THREADS"])

import numpy as np
import matplotlib.pyplot as plt
import sys

from qiskit_aer.noise import NoiseModel, depolarizing_error

# Determine the correct path for the 'src' directory
if '__file__' in globals():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
else:
    src_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))

sys.path.append(src_dir)

from constants import hbar, eV, G_F
from activate_backend import activate_backends, build_backend
from time_evolution import run_time_evolution
from observables import (
    PositionObservable,
    MomentumObservable,
    SigmaObservable,
    DirectSigmaObservable,
    SingleSiteEntanglementObservable,
    TwoSiteEntanglementObservable,
    BipartiteEntanglementObservable,
    GlobalMagicObservable,
    DensityMatrixObservable,
)


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq
trotter_order = "second" #first, second
ibm_service, ionq_provider = activate_backends()


def initialize_parameters(ibm_service, ionq_provider):
    params = {}

    # total particles/sites/qubits, all electron-flavored neutrinos and antineutrinos
    params["N_sites"] = 4
    N_sites = params["N_sites"]

    # --- masses / mixing ---
    m1 = 0 * eV                 # 1st mass eigenstate (Richers 2021)
    m2 = 0 * eV                 # 2nd mass eigenstate (Richers 2021)
    params["delta_m_squared"] = (m2**2 - m1**2)  # erg^2, so omega = 0 here
    theta_nu = 1000             # mixing angle, only used to build B below

    # --- run controls ---
    params["shots"] = 1042
    params["trotter_steps"] = 2  # Trotter steps per time step
    params["trotter_order"] = trotter_order
    params["optimization_level"] = 1
    params["df"] = 1             # degrees of freedom for chi-square test
    params["backend_name"] = backend_name
    params["datadir"] = os.path.join(os.getcwd(), "datafiles")

    # Sampled sigmas plus, from the exact state, the noise-free sigmas and the
    # many-body quantities: entropy and magic per site, per site pair, and for the
    # half-system bipartition, then the global magic with its Pauli weights, and
    # finally the density matrix itself so a run can be re-analysed offline.
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
        SingleSiteEntanglementObservable(),
        TwoSiteEntanglementObservable(),
        BipartiteEntanglementObservable(),
        GlobalMagicObservable(),
        DensityMatrixObservable(),
    ]

    # --- time grid ---
    params["tau"] = 0.2 * hbar
    ttotal = 100 * params["tau"]
    params["times"] = np.arange(0, ttotal, params["tau"])

    # --- geometry / box ---
    L = 2.1e-11                  # cm, domain size (big box length)
    params["L"] = L
    params["dx"] = L             # box length at a site
    params["dp"] = L             # shape-function width (unused here, kept for the evolve args)
    params["shape_name"] = "none"
    params["geometric_name"] = "physical"
    params["periodic"] = True
    params["advection"] = False
    params["B_pert"] = None
    params["alpha"] = 0

    # ------------------------------------------------------------------
    # Realistic depolarizing error probabilities (as of 2024):
    #
    # When a qubit undergoes depolarization, it transitions to a completely mixed state,
    # represented by the maximally mixed density matrix ρ=𝐼/2, where I is the identity matrix.
    #
    # IBM (superconducting):
    #   - Single-qubit gates:   0.05% – 0.3%   (p_gate1 = 0.0005 – 0.003)
    #   - Two-qubit (CNOT):     0.4% – 2%      (p_gate2 = 0.004  – 0.02)
    #
    # Quantinuum/IonQ (trapped ion):
    #   - Single-qubit: <0.05% (p_gate1 = 0.0001 – 0.0005)
    #   - Two-qubit:   <0.1%  (p_gate2 = 0.0005 – 0.001)
    #
    # Rigetti/other superconducting:
    #   - Single-qubit: 0.05% – 0.5% (p_gate1 = 0.0005 – 0.005)
    #   - Two-qubit:    1% – 5%      (p_gate2 = 0.01 – 0.05)
    #
    # For generic simulation:
    #   - Use p_gate1 ≈ 0.001 (0.1%) for 1q, p_gate2 ≈ 0.01 (1%) for 2q gates.
    # ------------------------------------------------------------------

    # Realistic (IBM-like) depolarizing probabilities. Nothing in the pipeline
    # reads these, but they stay in params so run_parameters.json records the noise
    # level of the run -- the backend's repr does not show its noise model.
    params["p_gate1"] = 0  # 0.1% depolarizing noise on single-qubit gates
    params["p_gate2"] = 0   # 1% depolarizing noise on two-qubit gates

    error_gate1 = depolarizing_error(params["p_gate1"], 1) # single qubit gate error actingon 1 qubit simultaneously.
    error_gate2 = depolarizing_error(params["p_gate2"], 2) #two-qubit gate error acting on 2 qubits simulatenously, at once.

    noise_depol = NoiseModel()
    # Use the basis gates your circuit uses!
    noise_depol.add_all_qubit_quantum_error(error_gate1, ["x", "sx", "rz", "id"])  # Modern IBM basis gates
    noise_depol.add_all_qubit_quantum_error(error_gate2, ["cx"])

    print(noise_depol)

    # --- backend ---
    # density_matrix is the only method that holds the noisy state exactly: any
    # other applies the noise model by quantum trajectories, which would leave the
    # entropies and magic sampled rather than exact (the entanglement observables
    # assert on this).
    backend_options = dict(
        method="density_matrix",
        noise_model=noise_depol,
    )
    params["backend"] = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
        ibm_backend="least_busy",
    )

    # --- number densities -> particle counts per site ---
    mu = 1
    params["N"] = mu * np.full(N_sites, (params["dx"]**3) / (np.sqrt(2) * G_F * N_sites))

    # --- B field (same for all particles) ---
    # Standard flavor-isospin convention B = (sin 2θ, 0, -cos 2θ), shared by every
    # test. Δm² is 0 here, so omega is 0 and this term does not contribute.
    B = np.array([np.sin(2 * theta_nu), 0, -np.cos(2 * theta_nu)])
    params["B"] = B / np.linalg.norm(B)

    # --- positions: both flavors on the same grid ---
    N_sites_eachflavor = N_sites // 2

    def generate_x_array(N_sites_eachflavor, L):
        one_flavor = [
            (i - 0.5) * L / N_sites_eachflavor
            for i in range(1, N_sites_eachflavor + 1)
        ]
        return np.array(one_flavor + one_flavor)

    params["x"] = generate_x_array(N_sites_eachflavor, L)

    # --- momenta: directions drawn from exp(-2*(vz-1)^2), peaked along +z ---
    v_list = []
    np.random.seed(42)  # Set fixed random seed for reproducibility
    for _ in range(N_sites):
        # Draw vz from a truncated Gaussian-like distribution proportional to
        # exp(-2*(vz-1)^2) # Rejection Sampling for vz
        accepted = False  #accepted is a boolean flag that keeps trying new samples until a vz is drawn according to the desired probability distribution.
        while not accepted:
            vz = np.random.uniform(-1, 1)  # Random vz between -1 and 1
            p = np.exp(-2 * (vz - 1)**2) # Gaussian distribution for probabilities , peaks at vz=1
            if np.random.uniform(0, 1) < p:  # Accept vz with probability p
                accepted = True
        # phi uniform in [0, 2pi]
        phi = np.random.uniform(0, 2 * np.pi)
        # Compute vx, vy
        vx = np.sqrt(max(0, 1 - vz**2)) * np.cos(phi)
        vy = np.sqrt(max(0, 1 - vz**2)) * np.sin(phi)
        v_list.append([vx, vy, vz])

    v = np.array(v_list)  # shape (N_sites, 3)
    for i in range(N_sites):
        v[i, :] /= np.linalg.norm(v[i, :])
    print("Velocity vector v:\n", v)
    params["p"] = v

    params["energy_sign"] = np.ones(N_sites)
    params["bit_list"] = ['0'] * (N_sites // 2 + 1) + ['1'] * (N_sites - N_sites // 2 - 1)

    return params


def simulate(params):
    datadir = params["datadir"]

    # Driver: sorts by position, builds the circuit, and each step calls the
    # recorders in params["observables"] -- the sampled and exact sigmas, the
    # entropies and magic -- plus the position/momentum files.
    run_time_evolution(params)

    def load_cols(name):
        return np.atleast_2d(np.loadtxt(os.path.join(datadir, name)))

    return {
        "times": load_cols("t_sigma_z.dat")[:, 0],
        "sigma_x": load_cols("t_sigma_x.dat")[:, 1:],
        "sigma_y": load_cols("t_sigma_y.dat")[:, 1:],
        "sigma_z": load_cols("t_sigma_z.dat")[:, 1:],
        "sigma_x_direct": load_cols("t_sigma_x_direct.dat")[:, 1:],
        "sigma_y_direct": load_cols("t_sigma_y_direct.dat")[:, 1:],
        "sigma_z_direct": load_cols("t_sigma_z_direct.dat")[:, 1:],
        "x_values": load_cols("t_xsiteval.dat")[:, 1:],
        "px_values": load_cols("t_pxsiteval.dat")[:, 1:],
        "renyi2_single": load_cols("t_renyi2_single_direct.dat")[:, 1:],
        "renyi2_two_site": load_cols("t_renyi2_two_site_direct.dat")[:, 1:],
        "renyi2_bipartite": load_cols("t_renyi2_bipartite_direct.dat")[:, 1],
        "magic_single": load_cols("t_magic_single_direct.dat")[:, 1:],
        "magic_two_site": load_cols("t_magic_two_site_direct.dat")[:, 1:],
        "magic_bipartite": load_cols("t_magic_bipartite_direct.dat")[:, 1],
        "magic_global": load_cols("t_magic_global_direct.dat")[:, 1],
        "pauli_weights": load_cols("t_pauli_weight_distribution_direct.dat")[:, 1:],
        "two_site_pairs": np.atleast_2d(
            np.loadtxt(os.path.join(datadir, "two_site_pairs.dat"), dtype=int)
        ),
    }


def plot_results(results, params):
    plotdir = os.path.join(os.getcwd(), "plots")
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    N_sites = params["N_sites"]
    times = results["times"] / hbar
    two_site_pairs = results["two_site_pairs"]

    def plot_all_sites(data, ylabel, title, filename, semilogy=False):
        plt.figure()
        for site in range(N_sites):
            y = data[:, site]
            if semilogy:
                plt.semilogy(times, y, label=f"site {site+1}")
            else:
                plt.plot(times, y, label=f"site {site+1}")
        plt.xlabel("Time/ ħ")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plotdir, filename), bbox_inches="tight")
        plt.show()

    def plot_all_pairs(data, pair_list, ylabel, title, filename, semilogy=False, ncol=2):
        plt.figure()

        for col, (i, j) in enumerate(pair_list):
            y = data[:, col]

            if semilogy:
                y = np.maximum(np.abs(y), 1e-300)
                plt.semilogy(times, y, label=f"({i+1},{j+1})")
            else:
                plt.plot(times, y, label=f"({i+1},{j+1})")

        plt.xlabel("Time/ ħ")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(ncol=ncol)
        plt.grid(True)
        plt.savefig(os.path.join(plotdir, filename), bbox_inches="tight")
        plt.show()

    def plot_scalar(data, ylabel, title, filename, label):
        plt.figure()
        plt.plot(times, data, label=label)
        plt.xlabel("Time/ ħ")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plotdir, filename), bbox_inches="tight")
        plt.show()

    # =========================
    # Sigma plots from counts
    # =========================
    for basis in ("x", "y", "z"):
        plot_all_sites(
            results[f"sigma_{basis}"],
            ylabel=fr"$\sigma_{basis}$",
            title=fr"$\sigma_{basis}$ vs Time for all sites ({N_sites} sites)",
            filename=f"Josh_sigma_{basis}_vs_time.pdf",
        )
        plot_all_sites(
            results[f"sigma_{basis}_direct"],
            ylabel=fr"$\sigma_{basis}$ (direct state)",
            title=fr"$\sigma_{basis}$ (direct state) vs Time for all sites ({N_sites} sites)",
            filename=f"Josh_sigma_{basis}_direct_vs_time.pdf",
        )

    # =========================
    # Single-site Rényi-2
    # =========================
    plot_all_sites(
        results["renyi2_single"],
        ylabel=r"$S_2$",
        title=fr"Single-site 2nd R\'enyi entropy vs Time ({N_sites} sites)",
        filename="Josh_renyi2_single_direct_vs_time.pdf"
    )

    # =========================
    # Two-site Rényi-2
    # =========================
    plot_all_pairs(
        results["renyi2_two_site"],
        two_site_pairs,
        ylabel=r"$S_2$",
        title=fr"Two-site 2nd R\'enyi entropy vs Time ({N_sites} sites)",
        filename="Josh_renyi2_two_site_direct_vs_time.pdf",
        ncol=2
    )

    # =========================
    # Bipartite Rényi-2
    # =========================
    plot_scalar(
        results["renyi2_bipartite"],
        ylabel=r"$S_2$",
        title=fr"Bipartite 2nd R\'enyi entropy vs Time ({N_sites} sites)",
        filename="Josh_renyi2_bipartite_direct_vs_time.pdf",
        label="bipartition",
    )

    # =========================
    # Single-site magic
    # =========================
    plot_all_sites(
        results["magic_single"],
        ylabel=r"$M_2$",
        title=fr"Single-site stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_single_direct_vs_time.pdf"
    )

    # =========================
    # Two-site magic
    # =========================
    plot_all_pairs(
        results["magic_two_site"],
        two_site_pairs,
        ylabel=r"$M_2$",
        title=fr"Two-site stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_two_site_direct_vs_time.pdf",
        ncol=2
    )

    # =========================
    # Bipartite magic
    # =========================
    plot_scalar(
        results["magic_bipartite"],
        ylabel=r"$M_2$",
        title=fr"Bipartite stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_bipartite_direct_vs_time.pdf",
        label="bipartition",
    )

    # =========================
    # Global magic
    # =========================
    plot_scalar(
        results["magic_global"],
        ylabel=r"$M_2$",
        title=fr"Global stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_global_direct_vs_time.pdf",
        label="global",
    )

    # =========================
    # Pauli weight distribution
    # =========================
    pauli_weights = results["pauli_weights"]
    max_weight = pauli_weights.shape[1] - 1

    plt.figure()
    for w in range(max_weight + 1):
        plt.plot(times, pauli_weights[:, w], label=fr"$w={w}$")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$A_w(t)$")
    plt.title(fr"Pauli weight distribution vs Time ({N_sites} qubits)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_pauli_weight_distribution_vs_time.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Pauli weight distribution, semilogy
    # =========================
    plt.figure()
    for w in range(max_weight + 1):
        y = np.maximum(np.abs(pauli_weights[:, w]), 1e-300)
        plt.semilogy(times, y, label=fr"$w={w}$")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$A_w(t)$")
    plt.title(fr"Pauli weight distribution vs Time, log scale ({N_sites} qubits)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_pauli_weight_distribution_vs_time_log.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Average Pauli weight
    # =========================
    weights = np.arange(max_weight + 1, dtype=float)
    norm = np.maximum(np.sum(pauli_weights, axis=1), 1e-300)
    average_pauli_weight = np.sum(pauli_weights * weights[None, :], axis=1) / norm

    plot_scalar(
        average_pauli_weight,
        ylabel=r"$\langle w \rangle$",
        title=fr"Average Pauli weight vs Time ({N_sites} qubits)",
        filename="Josh_average_pauli_weight_vs_time.pdf",
        label=r"$\langle w \rangle$",
    )

    # =========================
    # Position evolution
    # =========================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.ylabel("Time/ ħ")
    plt.xlabel("Position (x)")
    for site in range(N_sites):
        plt.plot(results["x_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_Particles_position_evolution.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Momentum evolution
    # =========================
    plt.figure()
    plt.title(f"Momentum Evolution for {N_sites} sites")
    plt.ylabel("Time/ ħ")
    plt.xlabel(r"Momentum in x direction ($p_x$)")
    for site in range(N_sites):
        plt.plot(results["px_values"][:, site], times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_Particles_momentum_evolution.pdf"), bbox_inches="tight")
    plt.show()


params = initialize_parameters(ibm_service, ionq_provider)
results = simulate(params)
# chi_square_value, *_ = chi_square_test(params["N_sites"], params["times"][-1], params["shots"])
plot_results(results, params)
