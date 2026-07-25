#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, Operator, random_hermitian, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2
from qiskit.quantum_info import partial_trace
from qiskit.circuit.library import StatePreparation
import sys
import os

#===========================#
# Limit the number of cores #
#===========================#
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
print("MKL_NUM_THREADS:", os.environ["MKL_NUM_THREADS"])
print("JULIA_NUM_THREADS:", os.environ["JULIA_NUM_THREADS"])

# Determine the correct path for the 'src' directory
if '__file__' in globals():
    # Running as a script
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
else:
    # Running in an interactive environment like Jupyter Notebook
    src_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))

sys.path.append(src_dir)


#=======================#
# Import QCNO libraries #
#=======================#
from hamiltonian import construct_hamiltonian
from momentum import momentum
from base_circuit import initialize_base_circuit
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from evolve import apply_one_timestep_dynamic_positions, apply_qubit_permutation
from perturb import pert_circuit
from activate_backend import activate_backends, build_backend, backend_supports_direct_state
from time_evolution import run_time_evolution

ibm_service, ionq_provider = activate_backends()


# =========================
# Parameter lists (Julia equivalent)
# =========================
backend_name = "aer"
trotter_order= "first"

delta_omega_list = [-0.5, 0.0, 1.0, 0.5, 0.125]
N_sites_list     = [   4,   4,   4,  4,     4]

# Roggero table
Roggero_table = np.array([
    [-0.5, 0,    0,    1.8, 1.0],
    [ 0.0, 0,    2.10, 0,   0.36],
    [ 0.01,5.4,  0,   -8,   0.35],
    [ 0.05,2.58, 0,    0,   0.31],
    [ 0.125,1.65,0,    1.4, 0.32],
    [ 0.25,1.22, 0,    1.6, 0.41],
    [ 0.5, 1.06, 0,    1.4, 0.61],
    [ 1.0, 0.96, 0,    0,   0.99]
])

delta_omega_Rog = Roggero_table[:, 0]
a_t = Roggero_table[:, 1]
b_t = Roggero_table[:, 2]
c_t = Roggero_table[:, 3]
pmin = Roggero_table[:, 4]

# remove - and . from strings
def safe_float_string(x):
    s = str(x)
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


# extract the fit parameters corresponding to a particular value of delta_omega
def get_Roggero_fit_params(delta_omega):
    matches = np.where(delta_omega_Rog == delta_omega)[0]
    if len(matches) == 0:
        raise ValueError(f"delta_omega={delta_omega} not found in Roggero_table")
    idx = int(matches[0])
    return a_t[idx], b_t[idx], c_t[idx], idx

# find the first minimum to compare with Roggero results
def find_first_local_minima_index(arr):
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i
    return -1

#========================================================#
# Constant inputs (independent of N_sites / delta_omega) #
#========================================================#
params = {}
params["shots"] = 10420
params["measure"] = ["Z"]  # Pauli bases sampled each step (Rog needs only sigma_z)
params["trotter_steps"] = 5  # try comparing for larger trotter steps
params["optimization_level"] = 0

backend_options = dict(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=10000000,
    matrix_product_state_truncation_threshold=1e-16,
    mps_sample_measure_algorithm="mps_apply_measure",
)
backend, euler_basis, basis_gates, two_qubit_gate = build_backend(
    backend_name, ibm_service, ionq_provider, backend_options=backend_options,
)
params["backend"] = backend
params["backend_name"] = backend_name
params["euler_basis"] = euler_basis
params["basis_gates"] = basis_gates
params["two_qubit_gate"] = two_qubit_gate
params["trotter_order"] = trotter_order

params["tolerance"] = 5e-1
params["df"] = 1
params["tau"] = 0.05 * hbar
params["ttotal"] = 5.0 * hbar
params["times"] = np.arange(0.0, params["ttotal"] + params["tau"], params["tau"])

params["dx"] = 1e-3
params["L"] = 1.0
params["dp"] = params["L"]

params["shape_name"] = "none"
params["geometric_name"] = "none"
params["periodic"] = False
params["alpha"] = 0.0
params["B_pert"] = None
params["advection"] = False  # True/False

# Match Julia create_gates B = [sin(2θ), 0, -cos(2θ)].
theta_nu = 0.0
B = np.array([np.sin(2 * theta_nu), 0.0, -np.cos(2 * theta_nu)], dtype=float)
params["B"] = B / np.linalg.norm(B)


def initialize_parameters(N_sites, delta_omega):
    """Fill in the parameters that depend on N_sites / delta_omega.

    The constant inputs are set once above; this updates the global params dict
    with the per-case values and returns it.
    """
    params["N_sites"] = N_sites
    params["delta_omega"] = delta_omega
    params["delta_m_squared"] = float(delta_omega)

    mu = np.ones(N_sites, dtype=float)
    params["N"] = mu * ((params["dx"] ** 3) / (np.sqrt(2.0) * G_F * N_sites))

    params["x"] = np.zeros(N_sites, dtype=float)

    px_a = np.full(N_sites // 2, 0.5, dtype=float)
    px_b = np.full(N_sites // 2, 0.5, dtype=float)
    px = np.concatenate((px_a, px_b))
    p = np.column_stack((px, np.zeros(N_sites), np.zeros(N_sites)))
    params["p"] = p

    # Roggero Julia convention: first half +1, second half -1. Negated so the
    # shared omega = Δm²/(2|p|)*energy_sign reproduces the Roggero sign
    # (originally omega carried an explicit leading minus).
    params["energy_sign"] = -np.array(
        [1 if i < N_sites // 2 else -1 for i in range(N_sites)],
        dtype=int,
    )

    # Match Julia productMPS(s, N -> N <= N/2 ? "Dn" : "Up")
    params["bit_list"] = ['1' if i < N_sites // 2 else '0' for i in range(N_sites)]

    return params

def run_single_config(N_sites, delta_omega):
    # set up parameters for this config
    params = initialize_parameters(N_sites, delta_omega)

    # each Roggero case gets its own datafiles subfolder
    case_tag = f"N{N_sites}_dw{safe_float_string(delta_omega)}"
    datadir = os.path.join(os.getcwd(), "datafiles", case_tag)
    os.makedirs(datadir, exist_ok=True)

    # DO THE REAL WORK
    # loop over integration times; the driver measures the observables named in
    # params ("measure") and writes the standardized output files.
    run_time_evolution(params, datadir)

    # read sigma_z back from file for the post-loop fit
    times = np.asarray(params["times"], dtype=float)
    τ = params["tau"]
    tolerance = params["tolerance"]

    sz_path = os.path.join(datadir, "t_sigma_z.dat")
    sigma_z_values = np.atleast_2d(np.loadtxt(sz_path))[:, 1:]  # drop leading time column
    print(f"sigma_z_values = {sigma_z_values}")

    sigma_z1_values = sigma_z_values[:, 0]   # first column = site 1 over all times
    print(f"sigma_z1_values (first column of sigma_z_values) = {sigma_z1_values}")

    # Match Julia quantity for the minimum search: use -Sz
    minus_sigma_z1 = -sigma_z1_values

    # Keep smoothing ONLY for locating the scatter point, not for plotting
    if len(times) >= 4:
        spline_sz = UnivariateSpline(times, minus_sigma_z1, s=0.03)
        minus_sigma_z1_smooth = spline_sz(times)
    else:
        minus_sigma_z1_smooth = minus_sigma_z1.copy()

    i_first_local_min = find_first_local_minima_index(minus_sigma_z1_smooth)
    t_min = None
    if i_first_local_min != -1:
        print(f"Index of the first local minimum from smooth curve: {i_first_local_min}")
        t_min = τ * i_first_local_min - τ
    else:
        print("No local minimum found in the smooth array.")

    a_fit, b_fit, c_fit, rog_index = get_Roggero_fit_params(delta_omega)
    t_p_Rog = a_fit * np.log(N_sites) + b_fit * np.sqrt(N_sites) + c_fit

    print("delta_omega =", delta_omega)
    print("N_sites =", N_sites)
    print("t_p_Rog =", t_p_Rog)

    if t_min is not None:
        assert abs(t_min / hbar - t_p_Rog) < (τ / hbar) + tolerance, (
            "The time of the first minimum survival probability is not within the expected range."
        )
        print("Test passed.")

    # ---- plot ----
    plotdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plotdir, exist_ok=True)

    times_over_hbar = times / hbar
    t_min_over_hbar = None if t_min is None else t_min / hbar

    plt.figure(figsize=(8, 6))
    plt.plot(times_over_hbar, minus_sigma_z1, label=f"My_plot_for_N_sites{N_sites}")
    plt.xlabel("t / ħ")
    plt.ylabel("-sigma_z(t)")
    plt.title(f"Running main_Rog script for N_sites = {N_sites}, delta_omega = {delta_omega}")

    if i_first_local_min != -1:
        plt.scatter(
            [t_p_Rog],
            [minus_sigma_z1[i_first_local_min]],
            label="t_p_Rog",
            color="orange",
            zorder=5
        )
        plt.scatter(
            [t_min_over_hbar],
            [minus_sigma_z1[i_first_local_min]],
            label="My t_min",
            color="green",
            zorder=5
        )
    else:
        nearest_idx = np.abs(times_over_hbar - t_p_Rog).argmin()
        plt.scatter(
            [t_p_Rog],
            [minus_sigma_z1[nearest_idx]],
            label="t_p_Rog",
            color="orange",
            zorder=5
        )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"main_Rog_{case_tag}.pdf"))
    plt.show()

    return sigma_z1_values, sigma_z_values, t_min, t_p_Rog, i_first_local_min


for list_index in range(len(N_sites_list)):
    run_single_config(N_sites_list[list_index], delta_omega_list[list_index])

