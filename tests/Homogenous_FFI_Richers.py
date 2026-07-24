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
from base_circuit import initialize_base_circuit
from run_parameters import write_run_parameters
from sigma_statistics import calc_mean_and_sigma
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from evolve import apply_qubit_permutation, apply_one_timestep_dynamic_positions
from perturb import pert_circuit
from activate_backend import activate_backends, build_backend


# In[ ]:


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq_simulator(ideal simulator non-native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm 
####FYI : Fake manila works only for qubits: 4 or less since FakeManila is a 5-qubit device model.
trotter_order= "second" #first, second
ibm_service, ionq_provider = activate_backends()


# In[ ]:


def initialize_parameters(ibm_service, ionq_provider):
    N_sites_eachflavor= 1 # total sites/particles/qubits that are evenly spaced "for each (electron) flavor" 
    N_sites = 2* (N_sites_eachflavor) # total particles/sites/qubits for all neutrino and anti neutrino electron flavored 
    m1 = 0*eV #ergs #1st mass eigenstate of neutrino in Richers(2021)
    m2 =0.008596511*eV   #ergs #2nd mass eigenstate of neutrino in Richers(2021)
    delta_m_squared = (m2**2 - m1**2) # mass square difference # (erg^2)
 
    shots = 1042
    trotter_steps= 10 # Number of Trotter steps per time step
    optimization_level = 1

    backend_options = dict(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=1,
        matrix_product_state_truncation_threshold=1e-100,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    backend, euler_basis, basis_gates, two_qubit_gate = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
        ibm_backend="least_busy",
    )

    df = 1 # degrees of freedom for chisquare test
    τ =1.666e-4  # Time step
    # ttotal =30*τ # total time of evolution # sec #variable
    ttotal =1.666e-2   # evolving for a little more
    # times = np.arange(0, τ  + ttotal, τ)  # Define time steps
    times = np.arange(0, ttotal, τ)  # Define time steps
    
    L = 1e7 # cm # domain size # (aka big box length)
    n_νₑ = 2.92e24  # cm^-3 # number density of electron flavor neutrino
    # n_νₑ =0
    n_νₑ̄ =  n_νₑ # cm^-3 # number density of electron flavor antineutrino
    Eνₑ =  50*MeV # energy of all neutrinos (P.S the its negative is energy of all antineutrinos)
    Eνₑ̄ = -1 * Eνₑ # specific to my case only. Since all neutrinos have same energy, except in my case anti neutrinos are moving in opposite direction to give it a negative sign
    Δx = L  # length of the box of interacting neutrinos at a site in cm
    Δp = L # width of shape function # not being used in this test but defined to keep the evolve function arguments consistent.
    t1 = 0.013328000000000001 #choose initial time for growth rate calculation #variable, not being used in this test
    t2 = 0.014827400000000001  #choose final time for growth rate calculation #variable, not being used in this test
    theta_nu = 1.74532925E-8  #1e-6 degrees # mixing_angle # = 1.74532925E-8 radians 
    periodic = True

    V = L**3 # volume of the big box containing all sites/particles
    # Create an array of dimension N and fill it half with values of sites containing all electron neutrinos 
    # # and other half with sites containing electron anti-neutrino. 
    N_νₑ  = n_νₑ * V 
    N_1 = np.full(N_sites // 2, N_νₑ)
    N_νₑ̄  = n_νₑ̄ * V 
    N_2 = np.full(N_sites // 2, N_νₑ̄)

    # Concatenate the arrays to get the total number of neutrinos
    N = np.concatenate((N_1, N_2))
    # print("N =", N)
    # Create a B vector which would be same for all N particles
    B = np.array([-np.sin(2 * theta_nu), 0, np.cos(2 * theta_nu)])
    B = B / np.linalg.norm(B)

    # Function to generate x_array such that the first particle is at position L/(2*N_sites)
    # while subsequent particles are at a position incremental by L/N_sites (grid style)
    def generate_x_array(N_sites, L):
        return [(i - 0.5) * L / N_sites for i in range(1, N_sites + 1)]
    # Generate position arrays
    x = generate_x_array(N_sites, L)
    y = generate_x_array(N_sites, L)
    z = generate_x_array(N_sites, L)

    
    # Function to generate a momentum array in px direction
    # that depicts the energy of neutrinos and anti-neutrinos in opposing beams
    def generate_px_array(N_sites):
        half_N_sites = N_sites // 2
        return np.concatenate([np.full(half_N_sites, Eνₑ), np.full(half_N_sites, Eνₑ̄)])

    def generate_py_array(N_sites): # check
        half_N_sites = N_sites // 2
        return np.concatenate([np.zeros(half_N_sites), np.zeros(half_N_sites)])

    def generate_pz_array(N_sites): #check
        half_N_sites = N_sites // 2
        return np.concatenate([np.zeros(half_N_sites), np.zeros(half_N_sites)])

    # Generate momentum arrays
    px = generate_px_array(N_sites)
    py = generate_py_array(N_sites)
    pz = generate_pz_array(N_sites)

    # p matrix with numbers generated from the p_array for all components (x, y, z)
    p = np.column_stack((px, py, pz))

    # Create an array with the first half as 1 and the rest as -1
    energy_sign = np.array([-1 if i < N_sites // 2 else 1 for i in range(N_sites)])
    p_mod, p_hat = momentum(p, N_sites)
    # Calculate omega for each site
    omega = [delta_m_squared / (2 * p_mod[i]) * energy_sign[i] for i in range(N_sites)]
    print(omega)
    #modify this shape_name dict thing
    shape_name = "none"  # variable.  #Select a shape function based on the shape_name variable form the list defined in dictionary in shape_func file

    B_pert= None
    alpha=0
    geometric_name= "physical"
    bit_list = ['0' if s == -1 else '1' for s in energy_sign]
    advection = False #True/False
    return (N_sites, theta_nu, geometric_name,bit_list,omega, B, B_pert, alpha, N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ, shape_name , shots, trotter_steps, backend,backend_name, df, τ, times,ttotal,optimization_level, periodic,
    two_qubit_gate, euler_basis, basis_gates, advection)


# In[ ]:


# Function to find the value of rho_emu at a given time
def get_rho_at_time(time_array, rho_array, target_time):
    # Find the index of the closest time value in time_array
    idx = (np.abs(np.array(time_array) - target_time)).argmin()
    # Return the corresponding rho_emu value
    return rho_array[idx]


# In[ ]:




# In[ ]:


def get_counts_and_sigmas(qc_base, N_sites, backend,backend_name, optimization_level, shots,    two_qubit_gate,
    euler_basis,
    basis_gates, df=0):
    counts_z, isa_circuit_z = meas_counts(
        qc_base, 'Z', N_sites, backend, backend_name, optimization_level, shots,
        two_qubit_gate, euler_basis, basis_gates
    )
    counts_x, isa_circuit_x = meas_counts(
        qc_base, 'X', N_sites, backend, backend_name, optimization_level, shots,
        two_qubit_gate, euler_basis, basis_gates
    )
    counts_y, isa_circuit_y = meas_counts(
        qc_base, 'Y', N_sites, backend, backend_name, optimization_level, shots,
        two_qubit_gate, euler_basis, basis_gates
    )

    sigma_z, std_sigma_z = calc_mean_and_sigma(counts_z, shots, 'Z', N_sites, df=df)
    sigma_x, std_sigma_x = calc_mean_and_sigma(counts_x, shots, 'X', N_sites, df=df)
    sigma_y, std_sigma_y = calc_mean_and_sigma(counts_y, shots, 'Y', N_sites, df=df)

    # Reverse if your calc_mean_and_sigma returns Qiskit bitstring order
    sigma_x = np.asarray(sigma_x)[::-1]
    sigma_y = np.asarray(sigma_y)[::-1]
    sigma_z = np.asarray(sigma_z)[::-1]

    rho_ee_sites = (sigma_z + 1.0) / 2.0
    rho_mumu_sites = (-sigma_z + 1.0) / 2.0
    rho_emu_sites = 0.5 * np.sqrt(sigma_x**2 + sigma_y**2)

    return {
        "counts_z": counts_z,
        "counts_x": counts_x,
        "counts_y": counts_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "rho_ee": rho_ee_sites,
        "rho_mumu": rho_mumu_sites,
        "rho_emu": rho_emu_sites,
        "isa_circuit_z": isa_circuit_z,
        "isa_circuit_x": isa_circuit_x,
        "isa_circuit_y": isa_circuit_y,
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


def evolve_and_simulate(
    times, omega, energy_sign, delta_m_squared, Eνₑ, t1, t2,
    N, B, B_pert, alpha, Δx, Δp, shape_name, N_sites, x, p, L,
    theta_nu, geometric_name, bit_list, backend, backend_name, shots, τ, df, ttotal,
    optimization_level, periodic, trotter_steps, trotter_order,
    two_qubit_gate, euler_basis, basis_gates,advection
):
    # record the exact inputs of this run before anything mutates them
    import inspect as _inspect
    _av = _inspect.getargvalues(_inspect.currentframe())
    write_run_parameters("datafiles/run_parameters.json", {n: _av.locals[n] for n in _av.args})

    datadir = os.path.join(os.getcwd(), "datafiles")
    os.makedirs(datadir, exist_ok=True)

    sigma_x_values = []
    sigma_y_values = []
    sigma_z_values = []

    ρₑₑ_array = []
    ρ_μμ_array = []
    ρₑμ_array = []

    ρₑₑ_direct_array = []
    ρ_μμ_direct_array = []
    ρₑμ_direct_array = []

    x_values = []
    px_values = []

    def reorder_sorted_to_original(data_sorted, particle_ids):
        data_sorted = np.asarray(data_sorted)
        data_original = np.empty_like(data_sorted)
        for sorted_slot, pid in enumerate(particle_ids):
            data_original[pid] = data_sorted[sorted_slot]
        return data_original

    def get_rho_at_time(times_arr, rho_arr, target_t, site_index=0):
        times_arr = np.asarray(times_arr, dtype=float)
        rho_arr = np.asarray(rho_arr, dtype=float)
        idx = int(np.argmin(np.abs(times_arr - target_t)))
        print(f"Nearest time to target {target_t} is times[{idx}] = {times_arr[idx]}")
        return rho_arr[idx, site_index]

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
    bit_list = np.asarray(bit_list)[perm0].tolist()
    print("initial sorted bit_list =", bit_list)

    qc_base = initialize_base_circuit(
        n_qubits=N_sites,
        bit_list_sorted=bit_list,
        B_pert=B_pert,
        alpha=alpha
    )
    direct_ok = backend_supports_direct_state(backend)

    isa_circuit_x = None
    isa_circuit_y = None
    isa_circuit_z = None
    isa_circuit_z_first = None

    # open files once
    file_handles = {
        "x": open(os.path.join(datadir, "t_xsiteval.dat"), "w"),
        "px": open(os.path.join(datadir, "t_pxsiteval.dat"), "w"),
        "sx": open(os.path.join(datadir, "t_sigma_x.dat"), "w"),
        "sy": open(os.path.join(datadir, "t_sigma_y.dat"), "w"),
        "sz": open(os.path.join(datadir, "t_sigma_z.dat"), "w"),
        "rho_ee_counts": open(os.path.join(datadir, "t_rho_ee_from_counts.dat"), "w"),
        "rho_mumu_counts": open(os.path.join(datadir, "t_rho_mumu_from_counts.dat"), "w"),
        "rho_emu_counts": open(os.path.join(datadir, "t_rho_emu_from_counts.dat"), "w"),
        "rho_ee_direct": open(os.path.join(datadir, "t_rho_ee_direct.dat"), "w"),
        "rho_mumu_direct": open(os.path.join(datadir, "t_rho_mumu_direct.dat"), "w"),
        "rho_emu_direct": open(os.path.join(datadir, "t_rho_emu_direct.dat"), "w"),
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
            # 1) Measurement-based rho from counts
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
            isa_circuit_x = meas_data["isa_circuit_x"]
            isa_circuit_y = meas_data["isa_circuit_y"]

            if np.isclose(t, τ, rtol=0.0, atol=1e-15):
                isa_circuit_z_first = isa_circuit_z

            sigma_x_sorted = meas_data["sigma_x"]
            sigma_y_sorted = meas_data["sigma_y"]
            sigma_z_sorted = meas_data["sigma_z"]

            rho_ee_sorted = np.asarray(meas_data["rho_ee"])
            rho_mumu_sorted = np.asarray(meas_data["rho_mumu"])
            rho_emu_sorted = np.asarray(meas_data["rho_emu"])

            # reorder sorted-site arrays -> original physical-particle order
            sigma_x_original = reorder_sorted_to_original(sigma_x_sorted, particle_ids)
            sigma_y_original = reorder_sorted_to_original(sigma_y_sorted, particle_ids)
            sigma_z_original = reorder_sorted_to_original(sigma_z_sorted, particle_ids)

            rho_ee_original = reorder_sorted_to_original(rho_ee_sorted, particle_ids)
            rho_mumu_original = reorder_sorted_to_original(rho_mumu_sorted, particle_ids)
            rho_emu_original = reorder_sorted_to_original(rho_emu_sorted, particle_ids)

            sigma_x_values.append(sigma_x_original.copy())
            sigma_y_values.append(sigma_y_original.copy())
            sigma_z_values.append(sigma_z_original.copy())

            ρₑₑ_array.append(rho_ee_original.copy())
            ρ_μμ_array.append(rho_mumu_original.copy())
            ρₑμ_array.append(rho_emu_original.copy())

            write_row(file_handles["sx"], t, sigma_x_original)
            write_row(file_handles["sy"], t, sigma_y_original)
            write_row(file_handles["sz"], t, sigma_z_original)

            write_row(file_handles["rho_ee_counts"], t, rho_ee_original)
            write_row(file_handles["rho_mumu_counts"], t, rho_mumu_original)
            write_row(file_handles["rho_emu_counts"], t, rho_emu_original)

            print(f"t={t} sigma_x(original order) = {sigma_x_original}")
            print(f"t={t} sigma_y(original order) = {sigma_y_original}")
            print(f"t={t} sigma_z(original order) = {sigma_z_original}")
            print(f"t={t} rho_ee from sigmas      = {rho_ee_original}")
            print(f"t={t} rho_mumu from sigmas    = {rho_mumu_original}")
            print(f"t={t} rho_emu from sigmas     = {rho_emu_original}")

            # -------------------------------
            # 2) Direct rho from reduced density matrices
            # -------------------------------
            if direct_ok:
                _, density_matrix_z, statevector_z = get_direct_state(
                    qc_base, backend, optimization_level
                )

                rho_ee_direct_sites = []
                rho_mumu_direct_sites = []
                rho_emu_direct_sites = []

                for i in range(N_sites):
                    traced_out = list(range(0, i)) + list(range(i + 1, N_sites))
                    rdm = partial_trace(density_matrix_z, traced_out)

                    rho_ee_direct_sites.append(np.real(rdm.data[0, 0]))
                    rho_mumu_direct_sites.append(np.real(rdm.data[1, 1]))
                    rho_emu_direct_sites.append(np.abs(rdm.data[0, 1]))

                rho_ee_direct_sites = np.asarray(rho_ee_direct_sites)
                rho_mumu_direct_sites = np.asarray(rho_mumu_direct_sites)
                rho_emu_direct_sites = np.asarray(rho_emu_direct_sites)

                rho_ee_direct_original = reorder_sorted_to_original(rho_ee_direct_sites, particle_ids)
                rho_mumu_direct_original = reorder_sorted_to_original(rho_mumu_direct_sites, particle_ids)
                rho_emu_direct_original = reorder_sorted_to_original(rho_emu_direct_sites, particle_ids)

                ρₑₑ_direct_array.append(rho_ee_direct_original.copy())
                ρ_μμ_direct_array.append(rho_mumu_direct_original.copy())
                ρₑμ_direct_array.append(rho_emu_direct_original.copy())

                write_row(file_handles["rho_ee_direct"], t, rho_ee_direct_original)
                write_row(file_handles["rho_mumu_direct"], t, rho_mumu_direct_original)
                write_row(file_handles["rho_emu_direct"], t, rho_emu_direct_original)

                print(f"t={t} rho_ee direct   = {rho_ee_direct_original}")
                print(f"t={t} rho_mumu direct = {rho_mumu_direct_original}")
                print(f"t={t} rho_emu direct  = {rho_emu_direct_original}")
            else:
                nan_vec = np.full(N_sites, np.nan, dtype=float)
                ρₑₑ_direct_array.append(nan_vec.copy())
                ρ_μμ_direct_array.append(nan_vec.copy())
                ρₑμ_direct_array.append(nan_vec.copy())

                write_row(file_handles["rho_ee_direct"], t, nan_vec)
                write_row(file_handles["rho_mumu_direct"], t, nan_vec)
                write_row(file_handles["rho_emu_direct"], t, nan_vec)

            # force data to disk during evolution
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

    sigma_x_values = np.asarray(sigma_x_values, dtype=float)
    sigma_y_values = np.asarray(sigma_y_values, dtype=float)
    sigma_z_values = np.asarray(sigma_z_values, dtype=float)

    ρₑₑ_array = np.asarray(ρₑₑ_array, dtype=float)
    ρ_μμ_array = np.asarray(ρ_μμ_array, dtype=float)
    ρₑμ_array = np.asarray(ρₑμ_array, dtype=float)

    ρₑₑ_direct_array = np.asarray(ρₑₑ_direct_array, dtype=float)
    ρ_μμ_direct_array = np.asarray(ρ_μμ_direct_array, dtype=float)
    ρₑμ_direct_array = np.asarray(ρₑμ_direct_array, dtype=float)

    # growth rate from counts-based rho_emu
    del_t = abs(t2 - t1)
    eps = 1e-300

    rho_emu_at_t1 = max(get_rho_at_time(times, ρₑμ_array, t1, site_index=0), eps)
    rho_emu_at_t2 = max(get_rho_at_time(times, ρₑμ_array, t2, site_index=0), eps)
    Im_omega = (1.0 / del_t) * abs(np.log(rho_emu_at_t2 / rho_emu_at_t1))

    rho_emu_at_t1_direct = max(get_rho_at_time(times, ρₑμ_direct_array, t1, site_index=0), eps)
    rho_emu_at_t2_direct = max(get_rho_at_time(times, ρₑμ_direct_array, t2, site_index=0), eps)
    Im_omega_direct = (1.0 / del_t) * abs(np.log(rho_emu_at_t2_direct / rho_emu_at_t1_direct))

    analytic_growth_rate = (abs(delta_m_squared) / (2 * hbar * Eνₑ))

    print(f"Growth rate from counts-based rho_emu: Im(omega) = {Im_omega}")
    print(f"Growth rate from direct rho_emu: Im(omega) = {Im_omega_direct}")
    print(f"Analytic growth rate: Im(omega) = {analytic_growth_rate}")

    tolerance = 1e-1
    assert abs((Im_omega_direct - analytic_growth_rate) / analytic_growth_rate) < tolerance, \
        "Direct growth rate deviates from analytic result beyond tolerance"

    return (
        sigma_x_values, sigma_y_values, sigma_z_values,
        x_values, px_values,
        ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
        ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
        isa_circuit_z_first, Im_omega, Im_omega_direct, analytic_growth_rate
    )


# In[ ]:


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





# In[ ]:


def plot_results(
    times, shots,
    sigma_x_values, sigma_y_values, sigma_z_values,
    x_values, px_values,
    ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
    ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
    N_sites, ttotal, τ, trotter_steps, trotter_order
):
    plotdir = os.path.join(
        os.getcwd(), "plots"
    )
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    times = np.asarray(times, dtype=float)
    sigma_x_values = np.asarray(sigma_x_values, dtype=float)
    sigma_y_values = np.asarray(sigma_y_values, dtype=float)
    sigma_z_values = np.asarray(sigma_z_values, dtype=float)

    ρₑₑ_array = np.asarray(ρₑₑ_array, dtype=float)
    ρ_μμ_array = np.asarray(ρ_μμ_array, dtype=float)
    ρₑμ_array = np.asarray(ρₑμ_array, dtype=float)

    ρₑₑ_direct_array = np.asarray(ρₑₑ_direct_array, dtype=float)
    ρ_μμ_direct_array = np.asarray(ρ_μμ_direct_array, dtype=float)
    ρₑμ_direct_array = np.asarray(ρₑμ_direct_array, dtype=float)

    x_values = np.asarray(x_values, dtype=float)
    px_values = np.asarray(px_values, dtype=float)

    def plot_all_sites(data, ylabel, title, filename, semilogy=False):
        plt.figure()
        for site in range(N_sites):
            y = data[:, site]
            if semilogy:
                # y = np.maximum(np.abs(y), 1e-300)
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

    # =========================
    # Sigma plots from counts
    # =========================
    plot_all_sites(
        sigma_x_values,
        ylabel=r"$\sigma_x$",
        title=fr"$\sigma_x$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_sigma_x_vs_time.pdf",
        semilogy=False
    )

    plot_all_sites(
        sigma_y_values,
        ylabel=r"$\sigma_y$",
        title=fr"$\sigma_y$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_sigma_y_vs_time.pdf",
        semilogy=False
    )

    plot_all_sites(
        sigma_z_values,
        ylabel=r"$\sigma_z$",
        title=fr"$\sigma_z$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_sigma_z_vs_time.pdf",
        semilogy=False
    )

    # =========================
    # Rho plots from counts
    # =========================
    plot_all_sites(
        ρₑₑ_array,
        ylabel=r"$\rho_{ee}$",
        title=fr"$\rho_{{ee}}$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_ee_counts_vs_time.pdf",
        semilogy=True
    )

    plot_all_sites(
        ρ_μμ_array,
        ylabel=r"$\rho_{\mu\mu}$",
        title=fr"$\rho_{{\mu\mu}}$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_mumu_counts_vs_time.pdf",
        semilogy=True
    )

    plot_all_sites(
        ρₑμ_array,
        ylabel=r"$|\rho_{e\mu}|$",
        title=fr"$|\rho_{{e\mu}}|$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_emu_counts_vs_time.pdf",
        semilogy=True
    )

    # =========================
    # Direct rho plots
    # =========================
    plot_all_sites(
        ρₑₑ_direct_array,
        ylabel=r"$\rho_{ee}^{direct}$",
        title=fr"Direct $\rho_{{ee}}$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_ee_direct_vs_time.pdf",
        semilogy=True
    )

    plot_all_sites(
        ρ_μμ_direct_array,
        ylabel=r"$\rho_{\mu\mu}^{direct}$",
        title=fr"Direct $\rho_{{\mu\mu}}$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_mumu_direct_vs_time.pdf",
        semilogy=True
    )

    plot_all_sites(
        ρₑμ_direct_array,
        ylabel=r"$|\rho_{e\mu}^{direct}|$",
        title=fr"Direct $|\rho_{{e\mu}}|$ vs Time for all sites ({N_sites} sites)",
        filename="Homo_rho_emu_direct_vs_time.pdf",
        semilogy=True
    )

    # =========================
    # Position evolution
    # =========================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.xlabel("Time")
    plt.ylabel("Position (x)")
    for site in range(N_sites):
        plt.plot( x_values[:, site], times,label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Homo_Particles_position_evolution.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Momentum evolution
    # =========================
    plt.figure()
    plt.title(f"Momentum Evolution for {N_sites} sites")
    plt.xlabel("Time")
    plt.ylabel(r"Momentum in x direction ($p_x$)")
    for site in range(N_sites):
        plt.plot(px_values[:, site],times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Homo_Particles_momentum_evolution.pdf"), bbox_inches="tight")
    plt.show()


# In[ ]:


(N_sites, theta_nu, geometric_name,bit_list,omega, B, B_pert,alpha, N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ, shape_name , shots, trotter_steps, backend,backend_name, df, τ, times,ttotal,optimization_level, periodic,
    two_qubit_gate, euler_basis, basis_gates, advection)= initialize_parameters(ibm_service, ionq_provider)
sigma_x_values, sigma_y_values, sigma_z_values, x_values, px_values, ρₑₑ_array, ρ_μμ_array, ρₑμ_array, ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array, circuit, Im_omega, Im_omega_direct, analytic_growth_rate= evolve_and_simulate(times, omega,energy_sign, delta_m_squared, Eνₑ, t1,t2, N, B, B_pert,alpha,Δx,Δp, shape_name, N_sites,x, p, L, theta_nu,geometric_name, bit_list,backend,backend_name, shots, τ, df, ttotal, optimization_level,periodic, trotter_steps, trotter_order,
    two_qubit_gate, euler_basis, basis_gates, advection)
# chi_square_value, Sx_array_CC, sx_array_QC, sigma_xi_array_QC,  Sy_array_CC, sy_array_QC, sigma_yi_array_QC, Sz_array_CC, sz_array_QC, sigma_zi_array_QC = chi_square_test(N_sites, ttotal, shots)
plot_results(times,shots,sigma_x_values, sigma_y_values, sigma_z_values, x_values, px_values, ρₑₑ_array, ρ_μμ_array, ρₑμ_array,ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array, N_sites,ttotal, τ,trotter_steps, trotter_order)
circuit.draw(output="mpl")  #circuit for first timestep

