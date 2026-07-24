#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Limit the number of cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
print("MKL_NUM_THREADS:", os.environ["MKL_NUM_THREADS"])
print("JULIA_NUM_THREADS:", os.environ["JULIA_NUM_THREADS"])


# In[ ]:


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


# 

# In[ ]:


# Determine the correct path for the 'src' directory
if '__file__' in globals():
    # Running as a script
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
else:
    # Running in an interactive environment like Jupyter Notebook
    src_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))

sys.path.append(src_dir)


# In[ ]:


from hamiltonian import construct_hamiltonian
from momentum import momentum
from meas_counts import get_direct_state, meas_counts
from run_parameters import write_run_parameters
from sigma_statistics import calc_mean_and_sigma
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from evolve import apply_one_timestep_dynamic_positions, apply_qubit_permutation
from perturb import pert_circuit
from activate_backend import activate_backends, build_backend


# In[ ]:


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq(ideal simulator non-native gateset), ionq_native (ideal simulator native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm 
trotter_order= "second" # first, second

ibm_service, ionq_provider = activate_backends()


# In[ ]:


# =========================
# Parameter lists (Julia equivalent)
# =========================
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


# In[ ]:


def safe_float_string(x):
    s = str(x)
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


# In[ ]:


def get_Roggero_fit_params(delta_omega):
    matches = np.where(delta_omega_Rog == delta_omega)[0]
    if len(matches) == 0:
        raise ValueError(f"delta_omega={delta_omega} not found in Roggero_table")
    idx = int(matches[0])
    return a_t[idx], b_t[idx], c_t[idx], idx


# In[ ]:


def find_first_local_minima_index(arr):
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i
    return -1


# In[ ]:


def initialize_parameters(N_sites, delta_omega, ibm_service, ionq_provider):
    delta_m_squared = float(delta_omega)

    shots = 10420
    trotter_steps = 5  #try comparing for larger trotter steps 
    optimization_level = 0

    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=10000000,
        matrix_product_state_truncation_threshold=1e-16,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    backend, euler_basis, basis_gates, two_qubit_gate = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
    )

    tolerance = 5e-1
    df = 1
    τ = 0.05 * hbar
    ttotal = 5.0 * hbar
    times = np.arange(0.0, ttotal + τ, τ)

    Δx = 1e-3
    L = 1.0
    Δp = L

    theta_nu = 0.0
    shape_name = "none"
    geometric_name = "none"
    periodic = False
    alpha = 0.0
    B_pert = None

    mu = np.ones(N_sites, dtype=float)
    N = mu * ((Δx ** 3) / (np.sqrt(2.0) * G_F * N_sites))

    x = np.zeros(N_sites, dtype=float)

    px_a = np.full(N_sites // 2, 0.5, dtype=float)
    px_b = np.full(N_sites // 2, 0.5, dtype=float)
    px = np.concatenate((px_a, px_b))
    p = np.column_stack((px, np.zeros(N_sites), np.zeros(N_sites)))

    # Roggero Julia convention: first half +1, second half -1.
    energy_sign = np.array(
        [1 if i < N_sites // 2 else -1 for i in range(N_sites)],
        dtype=int,
    )

    # Match Julia create_gates:
    # omega[i] = (Δm² / (2 * |p_i|)) * energy_sign[i]
    p_mod = np.linalg.norm(p, axis=1)
    omega = np.where(
        p_mod != 0.0,
        -(delta_m_squared / (2.0 * p_mod)) * energy_sign.astype(float),
        0.0,
    )

    # Match Julia create_gates B = [sin(2θ), 0, -cos(2θ)].
    B = np.array([np.sin(2 * theta_nu), 0.0, -np.cos(2 * theta_nu)], dtype=float)
    B = B / np.linalg.norm(B)

    # Match Julia productMPS(s, N -> N <= N/2 ? "Dn" : "Up")
    bit_list = ['1' if i < N_sites // 2 else '0' for i in range(N_sites)]
    advection = False #True/False
    return (
        N_sites, theta_nu, geometric_name, bit_list, omega, B, B_pert, alpha, N,
        x, p, energy_sign, Δx, Δp, L, delta_m_squared, shape_name, shots,
        trotter_steps, backend, backend_name, tolerance, df, τ, times, ttotal,
        optimization_level, periodic, two_qubit_gate, euler_basis, basis_gates,
        advection
    )



# In[ ]:


def initialize_base_circuit(n_qubits, bit_list_sorted=None, B_pert=None, alpha=None):
    qc = QuantumCircuit(n_qubits, n_qubits)

    if bit_list_sorted is None:
        raise ValueError("bit_list_sorted must be provided")

    bitstr = ''.join(bit_list_sorted)
    print("init bit_list_sorted =", bit_list_sorted)
    print("init bitstr =", bitstr)

    prep = StatePreparation(bitstr[::-1])  # Qiskit endianness
    qc.append(prep, qargs=range(n_qubits))
    qc.barrier(label="after_init")

    if B_pert is not None:
        pert_circuit(qc, B_pert, n_qubits, alpha)
        qc.barrier(label="after_pert")

    return qc


# In[ ]:


def get_counts_and_sigmas(qc_base, N_sites, backend, backend_name, optimization_level, shots,
    two_qubit_gate,
    euler_basis,
    basis_gates, df=0):
    counts_z, isa_circuit_z = meas_counts(
        qc_base, 'Z', N_sites, backend, backend_name, optimization_level, shots,
        two_qubit_gate, euler_basis, basis_gates
    )
    sigma_z, _ = calc_mean_and_sigma(counts_z, shots, 'Z', N_sites, df=df)

    sigma_z = np.asarray(sigma_z)[::-1]

    return {
        "sigma_z": sigma_z,
        "isa_circuit_z": isa_circuit_z,
    }


# In[ ]:


def backend_supports_direct_state(backend, backend_name=None):
    if backend_name is not None:
        name = str(backend_name).lower()
        # only allow true Aer simulator backends for direct state access
        return "aer" in name

    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        name_attr = name_attr()
    if name_attr is None:
        name_attr = str(backend)

    name_attr = str(name_attr).lower()
    return "aer" in name_attr


# In[ ]:


def simulate(
    times, omega, energy_sign, x, Δp, L, shape_name, N_sites, theta_nu,
    geometric_name, bit_list, backend, backend_name, shots, τ, df, ttotal,
    optimization_level, N, B, B_pert, alpha, Δx, p, tolerance,
    trotter_steps, trotter_order, periodic, delta_omega,
    two_qubit_gate, euler_basis, basis_gates,advection
):
    # record the exact inputs of this run before anything mutates them
    import inspect as _inspect
    _av = _inspect.getargvalues(_inspect.currentframe())
    write_run_parameters("datafiles/run_parameters.json", {n: _av.locals[n] for n in _av.args})
    import os
    import numpy as np
    from scipy.interpolate import UnivariateSpline

    datadir = os.path.join(os.getcwd(), "datafiles")
    os.makedirs(datadir, exist_ok=True)

    sigma_z1_values = []
    sigma_z_values = []
    x_values = []
    px_values = []

    def reorder_sorted_to_original(data_sorted, particle_ids):
        data_sorted = np.asarray(data_sorted)
        data_original = np.empty_like(data_sorted)
        for sorted_slot, pid in enumerate(particle_ids):
            data_original[pid] = data_sorted[sorted_slot]
        return data_original

    def write_row(fhandle, t, values):
        row = np.concatenate(([t], np.asarray(values, dtype=float).ravel()))
        np.savetxt(fhandle, row[None, :], fmt="%.16e")

    particle_ids = np.arange(N_sites)

    # initial sort
    perm0 = np.argsort(x)
    x = np.asarray(x)[perm0].copy()
    p = np.asarray(p)[perm0].copy()
    N = np.asarray(N)[perm0].copy()
    omega = np.asarray(omega)[perm0].copy()
    energy_sign = np.asarray(energy_sign)[perm0].copy()
    particle_ids = particle_ids[perm0].copy()
    bit_list = np.asarray(bit_list)[perm0].tolist()
    print("initial sorted bit_list =", bit_list)
    print("initial energy_sign =", energy_sign)

    qc_base = initialize_base_circuit(
        n_qubits=N_sites,
        bit_list_sorted=bit_list,
        B_pert=B_pert,
        alpha=alpha
    )
    direct_ok = backend_supports_direct_state(backend)

    isa_circuit_z = None
    isa_circuit_z_first = None

    case_tag = f"N{N_sites}_dw{safe_float_string(delta_omega)}"

    # open files once
    file_handles = {
        "x": open(os.path.join(datadir, f"t_xsiteval_{case_tag}.dat"), "w"),
        "px": open(os.path.join(datadir, f"t_pxsiteval_{case_tag}.dat"), "w"),
        "sz": open(os.path.join(datadir, f"t_sigma_z_{case_tag}.dat"), "w"),
        "minus_sz1": open(os.path.join(datadir, f"minus_sigma_z1_{case_tag}.dat"), "w"),
    }

    try:
        for step_idx, t in enumerate(times):
            print(f"\niteration={step_idx}, time={t}")

            # -------------------------------
            # Store x and px in original order
            # -------------------------------
            x_original = reorder_sorted_to_original(x, particle_ids)
            px_original = reorder_sorted_to_original(p[:, 0], particle_ids)

            x_values.append(x_original.copy())
            px_values.append(px_original.copy())

            write_row(file_handles["x"], t, x_original)
            write_row(file_handles["px"], t, px_original)

            # -------------------------------
            # 1) Measurement-based sigma_z from counts
            # -------------------------------
            meas_data = get_counts_and_sigmas(
                qc_base=qc_base,
                N_sites=N_sites,
                backend=backend,
                backend_name=backend_name,
                optimization_level=optimization_level,
                shots=shots,
                two_qubit_gate=two_qubit_gate,
                euler_basis=euler_basis,
                basis_gates=basis_gates,
                df=df
            )

            isa_circuit_z = meas_data["isa_circuit_z"]

            if np.isclose(t, τ, rtol=0.0, atol=1e-15):
                isa_circuit_z_first = isa_circuit_z

            sigma_z_sorted = meas_data["sigma_z"]
            sigma_z_original = reorder_sorted_to_original(sigma_z_sorted, particle_ids)
            sigma_z_values.append(sigma_z_original.copy())

            print(f"t={t} sigma_z(original order) = {sigma_z_original}")

            write_row(file_handles["sz"], t, sigma_z_original)

            minus_sigma_z1_now = -float(sigma_z_original[0])
            np.savetxt(
                file_handles["minus_sz1"],
                np.array([[t, minus_sigma_z1_now]]),
                fmt="%.16e"
            )

            # force write during evolution
            for fh in file_handles.values():
                fh.flush()

            # stop after storing last point
            if step_idx == len(times) - 1:
                break

            qc_base, N, x, p, omega, particle_ids, energy_sign = apply_one_timestep_dynamic_positions(
                qc_base,
                τ,
                N, x, Δp, L, shape_name, omega, B,
                N_sites, Δx, p, geometric_name,
                trotter_steps, trotter_order, periodic,
                particle_ids,
                energy_sign,
                advection,
            )

    finally:
        for fh in file_handles.values():
            fh.close()

    # convert to arrays
    times = np.asarray(times, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    px_values = np.asarray(px_values, dtype=float)

    sigma_z_values = np.asarray(sigma_z_values, dtype=float)
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

    return (
        sigma_z1_values,
        sigma_z_values,
        t_min,
        t_p_Rog,
        i_first_local_min,
        isa_circuit_z_first,
    )


# In[ ]:


def plot_results(times, shots, sigma_z1_values, N_sites, ttotal, t_min, t_p_Rog, i_first_local_min, delta_omega):
    plotdir = os.path.join(
        os.getcwd(), "plots"
    )
    os.makedirs(plotdir, exist_ok=True)

    times_over_hbar = np.asarray(times) / hbar
    t_min_over_hbar = None if t_min is None else t_min / hbar
    minus_sigma_z1 = -np.asarray(sigma_z1_values)   # raw curve for plotting

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
    case_tag = f"N{N_sites}_dw{safe_float_string(delta_omega)}"
    plt.savefig(
        os.path.join(plotdir, f"main_Rog_{case_tag}.pdf")
    )
    plt.show()



# In[ ]:


def main(N_sites, delta_omega):
    case_tag = f"N{N_sites}_dw{safe_float_string(delta_omega)}"
    (
        N_sites, theta_nu, geometric_name, bit_list, omega, B, B_pert, alpha, N,
        x, p, energy_sign, Δx, Δp, L, delta_m_squared, shape_name, shots,
        trotter_steps, backend, backend_name, tolerance, df, τ, times, ttotal,
        optimization_level, periodic,
        two_qubit_gate, euler_basis, basis_gates,advection
    ) = initialize_parameters(N_sites, delta_omega, ibm_service, ionq_provider)

    (
        sigma_z1_values,
        sigma_z_values,
        t_min,
        t_p_Rog,
        i_first_local_min,
        circuit
    ) = simulate(
        times=times,
        omega=omega,
        energy_sign=energy_sign,
        x=x,
        Δp=Δp,
        L=L,
        shape_name=shape_name,
        N_sites=N_sites,
        theta_nu=theta_nu,
        geometric_name=geometric_name,
        bit_list=bit_list,
        backend=backend,
        backend_name=backend_name,
        shots=shots,
        τ=τ,
        df=df,
        ttotal=ttotal,
        optimization_level=optimization_level,
        N=N,
        B=B,
        B_pert=B_pert,
        alpha=alpha,
        Δx=Δx,
        p=p,
        tolerance=tolerance,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
        periodic=periodic,
        delta_omega=delta_omega,
        two_qubit_gate=two_qubit_gate,
        euler_basis=euler_basis,
        basis_gates=basis_gates,
        advection=advection,
        
    )

    plot_results(
        times=times,
        shots=shots,
        sigma_z1_values=sigma_z1_values,
        N_sites=N_sites,
        ttotal=ttotal,
        t_min=t_min,
        t_p_Rog=t_p_Rog,
        i_first_local_min=i_first_local_min,
        delta_omega=delta_omega
    )


    return sigma_z1_values, sigma_z_values, t_min, t_p_Rog, i_first_local_min, circuit



# In[ ]:


for list_index in range(len(N_sites_list)):
    main(N_sites_list[list_index], delta_omega_list[list_index])

