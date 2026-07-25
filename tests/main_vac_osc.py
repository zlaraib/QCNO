#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
# Determine the correct path for the 'src' directory
if "__file__" in globals():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
else:
    src_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

from meas_counts import meas_counts
from base_circuit import initialize_base_circuit
from run_parameters import write_run_parameters
from sigma_statistics import calc_mean_and_sigma
from constants import hbar, eV, MeV, G_F
from evolve import apply_one_timestep_dynamic_positions
from activate_backend import activate_backends, build_backend


# In[ ]:


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq_simulator(ideal simulator non-native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm 
####FYI : Fake manila works only for qubits: 4 or less since FakeManila is a 5-qubit device model.
trotter_order= "first" #first, second
save_plots_flag = False

ibm_service, ionq_provider = activate_backends()


# In[ ]:


def initialize_parameters(ibm_service, ionq_provider):
    N_sites = 4
    L = 1.0
    ttotal = 1.94e-4
    tolerance = 5e-2
    shots = 1000
    trotter_steps = 1
    optimization_level = 1
    df = 0
    tau = 3.88e-6 # (ttotal/50 from before)
    times = np.arange(0.0, ttotal + 0.5 * tau, tau)
    
    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=1,
        matrix_product_state_truncation_threshold=1e-100,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    backend, euler_basis, basis_gates, two_qubit_gate = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
    )

    Delta_x = 1e-3
    Delta_p = L
    theta_nu = np.pi / 4
    shape_name = "none"
    geometric_name = "physical"
    periodic = False
 
    # Optional perturbation parameters
    B_pert = None
    alpha = None

    # Julia:
    # m1 = 0 * eV
    # m2 = 0.008596511 * eV
    m1 = 0.0 * eV
    m2 = 0.008596511 * eV
    delta_m_squared = abs(m2**2 - m1**2)

    # All sites are neutrinos
    energy_sign = np.ones(N_sites)


    p = np.ones((N_sites, 3), dtype=float) * MeV
    p_mod = np.linalg.norm(p, axis=1)


    mu = np.zeros(N_sites, dtype=float)
    N = mu * np.full(N_sites, (Delta_x**3) / (np.sqrt(2) * G_F), dtype=float)

    B = np.array([np.sin(2 * theta_nu), 0.0, -np.cos(2 * theta_nu)], dtype=float)
    B = B / np.linalg.norm(B)


    x0 = np.random.rand()
    y0 = np.random.rand()
    x = np.full(N_sites, x0, dtype=float)
    y = np.full(N_sites, y0, dtype=float)
    z = np.zeros(N_sites, dtype=float)


    omega = (delta_m_squared / (2.0 * p_mod)) * energy_sign 

    # Initial product state: first half Dn, second half Up
    # Use strings because later we join them into a bitstring
    initial_bits = ["1" if i < N_sites // 2 else "0" for i in range(N_sites)]
    
    advection = False #True/False
    return {
        "N_sites": N_sites,
        "theta_nu": theta_nu,
        "omega": omega,
        "B": B,
        "N": N,
        "x": x,
        "y": y,
        "z": z,
        "p": p,
        "p_mod": p_mod,
        "energy_sign": energy_sign,
        "Delta_x": Delta_x,
        "Delta_p": Delta_p,
        "L": L,
        "delta_m_squared": delta_m_squared,
        "shape_name": shape_name,
        "geometric_name": geometric_name,
        "periodic": periodic,
        "shots": shots,
        "trotter_steps": trotter_steps,
        "backend": backend,
        "backend_name": backend_name,
        "tolerance": tolerance,
        "tau": tau,
        "times": times,
        "ttotal": ttotal,
        "optimization_level": optimization_level,
        "initial_bits": initial_bits,
        "trotter_order": trotter_order,
        "df": df,
        "B_pert": B_pert,
        "alpha": alpha,
        "euler_basis": euler_basis,
        "basis_gates": basis_gates,
        "two_qubit_gate": two_qubit_gate,
        "advection": advection,
    }



# In[ ]:




# In[ ]:


def simulate(params):
    # record the exact inputs of this run before anything mutates them
    write_run_parameters("datafiles/run_parameters.json", params)

    times = params["times"]
    omega = params["omega"]
    N_sites = params["N_sites"]
    backend = params["backend"]
    backend_name_local = params["backend_name"]
    shots = params["shots"]
    tolerance = params["tolerance"]
    optimization_level = params["optimization_level"]
    N = params["N"]
    B = params["B"]
    Delta_x = params["Delta_x"]
    Delta_p = params["Delta_p"]
    p = params["p"]
    x = params["x"]
    L = params["L"]
    shape_name = params["shape_name"]
    geometric_name = params["geometric_name"]
    periodic = params["periodic"]
    trotter_steps = params["trotter_steps"]
    trotter_order_local = params["trotter_order"]
    df = params["df"]
    tau = params["tau"]
    initial_bits = params["initial_bits"]
    B_pert = params["B_pert"]
    alpha = params["alpha"]
    two_qubit_gate = params["two_qubit_gate"]
    euler_basis = params["euler_basis"]
    basis_gates = params["basis_gates"]
    energy_sign = params["energy_sign"]
    advection = params["advection"]
    
    
    datadir = os.path.join(os.getcwd(), "datafiles")
    os.makedirs(datadir, exist_ok=True)

    def reorder_sorted_to_original(data_sorted, particle_ids):
        data_sorted = np.asarray(data_sorted)
        data_original = np.empty_like(data_sorted)
        for sorted_slot, pid in enumerate(particle_ids):
            data_original[pid] = data_sorted[sorted_slot]
        return data_original
    
    sigma_x_all_times = []
    sigma_y_all_times = []
    sigma_z_all_times = []

    x_values = []
    px_values = []

    sigma_z_site1_values = []
    Sz_site1_values = []

    expected_sigma_z_site1 = []
    expected_Sz_site1 = []

    last_circuit = None

    def write_row(fhandle, t, values):
        row = np.concatenate(([t], np.asarray(values, dtype=float).ravel()))
        np.savetxt(fhandle, row[None, :], fmt="%.16e")

    # original physical labels
    particle_ids = np.arange(N_sites)

    # initial sort
    perm0 = np.argsort(x)
    x = np.asarray(x)[perm0].copy()
    p = np.asarray(p)[perm0].copy()
    N = np.asarray(N)[perm0].copy()
    omega = np.asarray(omega)[perm0].copy()
    energy_sign = np.asarray(energy_sign)[perm0].copy()
    particle_ids = particle_ids[perm0].copy()
    initial_bits = np.asarray(initial_bits)[perm0].tolist()
    print("initial sorted bit_list =", initial_bits)

    print("one cycle time:", 2.0 * np.pi / omega[0])

    qc_base = initialize_base_circuit(
        n_qubits=N_sites,
        bit_list_sorted=initial_bits,
        B_pert=B_pert,
        alpha=alpha,
    )

    file_handles = {
        "sigma_x": open(os.path.join(datadir, "t_sigma_x.dat"), "w"),
        "sigma_y": open(os.path.join(datadir, "t_sigma_y.dat"), "w"),
        "sigma_z": open(os.path.join(datadir, "t_sigma_z.dat"), "w"),
        "sigma_z_site1": open(os.path.join(datadir, "t_sigma_z_site1.dat"), "w"),
        "Sz_site1": open(os.path.join(datadir, "t_Sz_site1.dat"), "w"),
        "expected_sigma_z_site1": open(os.path.join(datadir, "t_expected_sigma_z_site1.dat"), "w"),
        "expected_Sz_site1": open(os.path.join(datadir, "t_expected_Sz_site1.dat"), "w"),
    }

    try:
        for step_idx, t in enumerate(times):
            expected_sz = -0.5 * np.cos(omega[0] / hbar * t)
            expected_sigma_z = 2.0 * expected_sz

            expected_Sz_site1.append(expected_sz)
            expected_sigma_z_site1.append(expected_sigma_z)
            
            counts_z, isa_circuit_z = meas_counts(
                qc_base, "Z", N_sites, backend, backend_name_local, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates,
            )
            counts_x, isa_circuit_x = meas_counts(
                qc_base, "X", N_sites, backend, backend_name_local, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates,
            )
            counts_y, isa_circuit_y = meas_counts(
                qc_base, "Y", N_sites, backend, backend_name_local, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates,
            )

            sigma_z, _ = calc_mean_and_sigma(counts_z, shots, N_sites, df=df)
            sigma_x, _ = calc_mean_and_sigma(counts_x, shots, N_sites, df=df)
            sigma_y, _ = calc_mean_and_sigma(counts_y, shots, N_sites, df=df)

            sigma_x = np.asarray(sigma_x, dtype=float)[::-1]
            sigma_y = np.asarray(sigma_y, dtype=float)[::-1]
            sigma_z = np.asarray(sigma_z, dtype=float)[::-1]

            sigma_x_all_times.append(sigma_x.copy())
            sigma_y_all_times.append(sigma_y.copy())
            sigma_z_all_times.append(sigma_z.copy())

            sigma_z_site1 = float(sigma_z[0])
            Sz_site1 = 0.5 * sigma_z_site1

            sigma_z_site1_values.append(sigma_z_site1)
            Sz_site1_values.append(Sz_site1)
            last_circuit = isa_circuit_z

            print(f"t = {t:.8e}, sigma_z_site1 = {sigma_z_site1:.8e}, <Sz>_site1 = {Sz_site1:.8e}")

            write_row(file_handles["sigma_x"], t, sigma_x)
            write_row(file_handles["sigma_y"], t, sigma_y)
            write_row(file_handles["sigma_z"], t, sigma_z)
            np.savetxt(file_handles["sigma_z_site1"], np.array([[t, sigma_z_site1]]), fmt="%.16e")
            np.savetxt(file_handles["Sz_site1"], np.array([[t, Sz_site1]]), fmt="%.16e")
            np.savetxt(file_handles["expected_sigma_z_site1"], np.array([[t, expected_sigma_z]]), fmt="%.16e")
            np.savetxt(file_handles["expected_Sz_site1"], np.array([[t, expected_sz]]), fmt="%.16e")

            for fh in file_handles.values():
                fh.flush()

            if step_idx == len(times) - 1:
                break
            # rename this to t_final instead of one_timestep
            qc_base, N, x, p, omega, particle_ids, energy_sign = apply_one_timestep_dynamic_positions(
                qc_base,
                tau,
                N, x, Delta_p, L, shape_name, omega, B,
                N_sites, Delta_x, p, geometric_name,
                trotter_steps, trotter_order, periodic,
                particle_ids,
                energy_sign,
                advection,
            )

    finally:
        for fh in file_handles.values():
            fh.close()

    sigma_x_all_times = np.array(sigma_x_all_times, dtype=float)
    sigma_y_all_times = np.array(sigma_y_all_times, dtype=float)
    sigma_z_all_times = np.array(sigma_z_all_times, dtype=float)

    sigma_z_site1_values = np.array(sigma_z_site1_values, dtype=float)
    Sz_site1_values = np.array(Sz_site1_values, dtype=float)
    expected_sigma_z_site1 = np.array(expected_sigma_z_site1, dtype=float)
    expected_Sz_site1 = np.array(expected_Sz_site1, dtype=float)
    times = np.array(times, dtype=float)

    max_abs_err = np.max(np.abs(Sz_site1_values - expected_Sz_site1))
    print("max |<Sz>_site1 - analytic| =", max_abs_err)

    if backend_name_local == "aer":
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
        "circuit": last_circuit,
    }


# In[ ]:


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


# In[ ]:


params = initialize_parameters(ibm_service, ionq_provider)
results = simulate(params)
plot_results(results, params)

if results["circuit"] is not None:
    results["circuit"].draw(output="mpl")

