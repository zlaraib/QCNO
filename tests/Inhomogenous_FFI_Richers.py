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
from activate_backend import activate_backends, build_backend, backend_supports_direct_state


# In[ ]:


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq_simulator(ideal simulator non-native gateset), ionq_qpu (ionq device non-native gateset), ionq_noisy_sim (noisy simulator non-native gateset), ibm 
####FYI : Fake manila works only for qubits: 4 or less since FakeManila is a 5-qubit device model.
trotter_order= "second" #first, second

ibm_service, ionq_provider = activate_backends()


# In[ ]:


def initialize_parameters(ibm_service, ionq_provider):
    N_sites_eachflavor= 5 # total sites/particles/qubits that are evenly spaced "for each (electron) flavor" 
    N_sites = 2* (N_sites_eachflavor) # total particles/sites/qubits for all neutrino and anti neutrino electron flavored 
    m1 = 0.0 #ergs #1st mass eigenstate of neutrino in Richers(2021)
    m2 = 0   #ergs #2nd mass eigenstate of neutrino in Richers(2021)
    delta_m_squared = (m2**2 - m1**2) # mass square difference # (erg^2)
 
    shots = 1024
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
    )

    df = 1 # degrees of freedom for chisquare test
    τ = 5E-12  # Time step
    # ttotal =5* τ # total time to evolve the system for (e.g., 10 Trotter steps per time step)
    ttotal =1.75E-10  # evolving for a little more
    # ttotal =9E-11
    times = np.arange(0, ttotal, τ)  # Define time steps
    
    L = 1 # cm # domain size # (aka big box length)
    n_νₑ = 4.891290848285061e+32  # cm^-3 # number density of electron flavor neutrino
    # n_νₑ =0
    n_νₑ̄ =  n_νₑ # cm^-3 # number density of electron flavor antineutrino
    Eνₑ =  50*MeV # energy of all neutrinos (P.S the its negative is energy of all antineutrinos)
    Eνₑ̄ = -1 * Eνₑ # specific to my case only. Since all neutrinos have same energy, except in my case anti neutrinos are moving in opposite direction to give it a negative sign
    Δx = L/N_sites_eachflavor  # length of the box of interacting neutrinos at a site in cm
    Δp = Δx # width of shape function # not being used in this test but defined to keep the evolve function arguments consistent.
    # t1 = 2.1e-10 #choose initial time for growth rate calculation #variable, not being used in this test
    # t2 = 2.45e-10 #choose final time for growth rate calculation #variable, not being used in this test
    t1 = 6e-11 #choose initial time for growth rate calculation #variable, not being used in this test
    t2 = 1.5e-10 #choose final time for growth rate calculation #variable, not being used in this test
    theta_nu = 1.74532925E-8  #1e-6 degrees # mixing_angle # = 1.74532925E-8 radians 
    periodic = True

    V = Δx**3 # volume of the big box containing all sites/particles
    # Create an array of dimension N and fill it half with values of sites containing all electron neutrinos 
    # # and other half with sites containing electron anti-neutrino. 
    N_νₑ  = n_νₑ * V 
    N_1 = np.full(N_sites // 2, N_νₑ)
    N_νₑ̄  = n_νₑ̄ * V 
    N_2 = np.full(N_sites // 2, N_νₑ̄)

    # Concatenate the arrays to get the total number of neutrinos
    N = np.concatenate((N_1, N_2))
    print("N =", N)
    # Create a B vector which would be same for all N particles
    B = np.array([-np.sin(2 * theta_nu), 0, np.cos(2 * theta_nu)])
    B = B / np.linalg.norm(B)



    def generate_mu_x_array(N_sites_eachflavor, L):
        return [(i - 0.5) * L / N_sites_eachflavor for i in range(1, N_sites_eachflavor + 1)]

    def generate_antimu_x_array(N_sites_eachflavor, L):
        return [(i - 0.5) * L / N_sites_eachflavor for i in range(1, N_sites_eachflavor + 1)]
    
    def generate_x_array(N_sites_eachflavor, L):
        mu_x_array = generate_mu_x_array(N_sites_eachflavor, L)
        antimu_x_array = generate_antimu_x_array(N_sites_eachflavor, L)
        x_array = mu_x_array + antimu_x_array
        return x_array
    
    # Generate position arrays
    x = generate_x_array(N_sites_eachflavor, L)
    y = generate_x_array(N_sites_eachflavor, L)
    z = generate_x_array(N_sites_eachflavor, L)
    
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
    shape_name = "triangular"  # variable.  #Select a shape function based on the shape_name variable form the list defined in dictionary in shape_func file

    def generate_B_pert(theta_pert):
        """
        Generate the perturbation vector B_pert exactly like the Julia version:
        B_pert = [-sin(2*theta_pert), 0, cos(2*theta_pert)], normalized.
        """
        B_pert = np.array([
            
            -np.sin(2 * theta_pert),
            0,
            np.cos(2 * theta_pert),
        ])

        # Normalize to ensure unit vector
        B_pert /= np.linalg.norm(B_pert)
        return B_pert

    # Example usage
    theta_pert = theta_nu  # or whatever small perturbation you want
    B_pert = generate_B_pert(theta_pert)
    print("B_pert =", B_pert)
    alpha=1e-2
    print("Norm =", np.linalg.norm(B_pert))  
    geometric_name= "physical"
    advection = True #True/False
    bit_list = ['0' if s == -1 else '1' for s in energy_sign]
    return (
    N_sites, theta_nu, geometric_name, bit_list, omega, B, B_pert, alpha,
    N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ,
    shape_name, shots, trotter_steps, backend, backend_name, df, τ, times,
    ttotal, optimization_level, periodic,
    two_qubit_gate, euler_basis, basis_gates, advection,
)


# In[ ]:


# Function to find the value of rho_emu at a given time
def get_rho_at_time(time_array, rho_array, target_time):
    # Find the index of the closest time value in time_array
    idx = (np.abs(np.array(time_array) - target_time)).argmin()
    # Return the corresponding rho_emu value
    return rho_array[idx]


# In[ ]:




# In[ ]:


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





# In[ ]:


def evolve_and_simulate(
    times, omega, energy_sign, delta_m_squared, Eνₑ, t1, t2,
    N, B, B_pert, alpha, Δx, Δp, shape_name, N_sites, x, p, L,
    theta_nu, geometric_name, bit_list, backend, backend_name, shots, τ, df, ttotal,
    optimization_level, periodic, trotter_steps, trotter_order,
    two_qubit_gate, euler_basis, basis_gates,advection,
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

    # open files once and keep writing as evolution proceeds
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
            counts_z, isa_circuit_z = meas_counts(
                qc_base, 'Z', N_sites, backend, backend_name, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates
            )
            counts_x, isa_circuit_x = meas_counts(
                qc_base, 'X', N_sites, backend, backend_name, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates
            )
            counts_y, isa_circuit_y = meas_counts(
                qc_base, 'Y', N_sites, backend, backend_name, optimization_level,
                shots, two_qubit_gate, euler_basis, basis_gates
            )

            sigma_z_sorted, _ = calc_mean_and_sigma(counts_z, shots, N_sites, df=df)
            sigma_x_sorted, _ = calc_mean_and_sigma(counts_x, shots, N_sites, df=df)
            sigma_y_sorted, _ = calc_mean_and_sigma(counts_y, shots, N_sites, df=df)

            # Reverse if calc_mean_and_sigma returns Qiskit bitstring order
            sigma_x_sorted = np.asarray(sigma_x_sorted)[::-1]
            sigma_y_sorted = np.asarray(sigma_y_sorted)[::-1]
            sigma_z_sorted = np.asarray(sigma_z_sorted)[::-1]

            rho_ee_sorted = (sigma_z_sorted + 1.0) / 2.0
            rho_mumu_sorted = (-sigma_z_sorted + 1.0) / 2.0
            rho_emu_sorted = 0.5 * np.sqrt(sigma_x_sorted**2 + sigma_y_sorted**2)

            if np.isclose(t, τ, rtol=0.0, atol=1e-15):
                isa_circuit_z_first = isa_circuit_z

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

            # make sure rows are physically written to disk each iteration
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

    k = 2 * np.pi / L
    analytic_growth_rate = (abs(delta_m_squared) / (2 * hbar * Eνₑ)) + c * k

    # add the growth rate line on the plot also
    print(f"Growth rate from counts-based rho_emu: Im(omega) = {Im_omega}")
    print(f"Growth rate from direct rho_emu: Im(omega) = {Im_omega_direct}")
    print(f"Analytic growth rate: Im(omega) = {analytic_growth_rate}")

    return (
        sigma_x_values, sigma_y_values, sigma_z_values,
        x_values, px_values,
        ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
        ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
        isa_circuit_z_first, Im_omega, Im_omega_direct, analytic_growth_rate
    )


# In[ ]:


def plot_results(
    times, shots,
    sigma_x_values, sigma_y_values, sigma_z_values,
    x_values, px_values,
    ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
    ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
    N_sites, ttotal, τ, trotter_steps, trotter_order,
    backend=None,
    backend_name=None,
    analytic_growth_rate=None,
    fit_t1=None,
    fit_t2=None,
    analytic_offset=100.0,
):
    plotdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plotdir, exist_ok=True)

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

    eps = 1e-300

    def safe_semilogy_y(y):
        y = np.asarray(y, dtype=float)
        y = np.abs(y)
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

    # =========================
    # Sigma plots from counts
    # =========================
    plot_all_sites(
        sigma_x_values,
        ylabel=r"$\sigma_x$",
        title=fr"$\sigma_x$ vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_sigma_x_vs_time.pdf",
        semilogy=False,
    )

    plot_all_sites(
        sigma_y_values,
        ylabel=r"$\sigma_y$",
        title=fr"$\sigma_y$ vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_sigma_y_vs_time.pdf",
        semilogy=False,
    )

    plot_all_sites(
        sigma_z_values,
        ylabel=r"$\sigma_z$",
        title=fr"$\sigma_z$ vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_sigma_z_vs_time.pdf",
        semilogy=False,
    )

    # =========================
    # Rho plots from counts
    # =========================
    plot_all_sites(
        ρₑₑ_array,
        ylabel=r"$\rho_{ee}$",
        title=fr"$\rho_{{ee}}$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_rho_ee_counts_vs_time.pdf",
        semilogy=True,
    )

    plot_all_sites(
        ρ_μμ_array,
        ylabel=r"$\rho_{\mu\mu}$",
        title=fr"$\rho_{{\mu\mu}}$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_rho_mumu_counts_vs_time.pdf",
        semilogy=True,
    )

    plot_all_sites(
        ρₑμ_array,
        ylabel=r"$|\rho_{e\mu}|$",
        title=fr"$|\rho_{{e\mu}}|$ from counts vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_rho_emu_counts_vs_time.pdf",
        semilogy=True,
    )

    # =========================
    # Direct rho plots
    # =========================
    plot_all_sites(
        ρₑₑ_direct_array,
        ylabel=r"$\rho_{ee}^{direct}$",
        title=fr"Direct $\rho_{{ee}}$ vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_rho_ee_direct_vs_time.pdf",
        semilogy=True,
    )

    plot_all_sites(
        ρ_μμ_direct_array,
        ylabel=r"$\rho_{\mu\mu}^{direct}$",
        title=fr"Direct $\rho_{{\mu\mu}}$ vs Time for all sites ({N_sites} sites)",
        filename="Inhomo_rho_mumu_direct_vs_time.pdf",
        semilogy=True,
    )

    # =========================
    # Direct rho_emu plot + shifted analytic line
    # =========================
    plt.figure()

    for site in range(N_sites):
        y = safe_semilogy_y(ρₑμ_direct_array[:, site])
        plt.semilogy(times, y, label=f"site {site+1}")

    bond_dim = get_mps_bond_dimension_from_backend(backend)

    should_plot_analytic_line = (
        backend_name == "aer"
        and bond_dim == 1
        and analytic_growth_rate is not None
    )

    if should_plot_analytic_line:
        site_for_line = 0

        rho_target = safe_semilogy_y(ρₑμ_direct_array[:, site_for_line])

        mask_fit = (times >= fit_t1) & (times <= fit_t2)

        x_fit = times[mask_fit]
        y_fit = rho_target[mask_fit]

        mask_valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (y_fit > 0)

        if np.count_nonzero(mask_valid) >= 2:
            x_fit = x_fit[mask_valid]
            y_fit = y_fit[mask_valid]

            t_ref = x_fit[-1]

            ymax_data = np.nanmax(ρₑμ_direct_array)

            y_top =  analytic_offset * ymax_data

            y_analytic = (
                y_top
                * np.exp(
                    analytic_growth_rate * (x_fit - t_ref)
                )
            )

            plt.semilogy(
                x_fit,
                y_analytic,
                "--",
                color="gray",
                linewidth=2,
                label=fr"analytic growth: {analytic_growth_rate:.2e} s$^{{-1}}$",
            )

            mid_idx = len(x_fit) // 2

            plt.text(
                x_fit[mid_idx],
                y_analytic[mid_idx] * 6,
                fr"${analytic_growth_rate:.2e}\ \mathrm{{s}}^{{-1}}$",
                color="gray",
                fontsize=10,
                rotation=45,
                ha="center",
                va="center",
            )

            print(
                f"Plotted shifted analytic line from t={x_fit[0]:.3e} "
                f"to t={x_fit[-1]:.3e} with offset={analytic_offset}"
            )
        else:
            print(
                "Skipping analytic line: not enough positive finite "
                f"rho_emu_direct points between {fit_t1:.3e} and {fit_t2:.3e}."
            )

    plt.xlabel("Time")
    plt.ylabel(r"$|\rho_{e\mu}^{direct}|$")
    plt.title(fr"Direct $|\rho_{{e\mu}}|$ vs Time for all sites ({N_sites} sites)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, "Inhomo_rho_emu_direct_vs_time.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    # =========================
    # Position evolution
    # =========================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.xlabel("Position (x)")
    plt.ylabel("Time")

    for site in range(N_sites):
        plt.plot(x_values[:, site], times, label=f"Site {site+1}")

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, "Inhomo_Particles_position_evolution.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    # =========================
    # Momentum evolution
    # =========================
    plt.figure()
    plt.title(f"Momentum Evolution for {N_sites} sites")
    plt.xlabel(r"Momentum in x direction ($p_x$)")
    plt.ylabel("Time")

    for site in range(N_sites):
        plt.plot(px_values[:, site], times, label=f"Site {site+1}")

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, "Inhomo_Particles_momentum_evolution.pdf"),
        bbox_inches="tight",
    )
    plt.show()


# In[ ]:


(
    N_sites, theta_nu, geometric_name, bit_list, omega, B, B_pert, alpha,
    N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ,
    shape_name, shots, trotter_steps, backend, backend_name, df, τ, times,
    ttotal, optimization_level, periodic,
    two_qubit_gate, euler_basis, basis_gates,advection,
) = initialize_parameters(ibm_service, ionq_provider)

(
    sigma_x_values, sigma_y_values, sigma_z_values,
    x_values, px_values,
    ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
    ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
    circuit, Im_omega, Im_omega_direct, analytic_growth_rate
) = evolve_and_simulate(
    times, omega, energy_sign, delta_m_squared, Eνₑ, t1, t2,
    N, B, B_pert, alpha, Δx, Δp, shape_name, N_sites, x, p, L,
    theta_nu, geometric_name, bit_list, backend, backend_name, shots, τ, df, ttotal,
    optimization_level, periodic, trotter_steps, trotter_order,
    two_qubit_gate, euler_basis, basis_gates,advection,
)
# chi_square_value, Sx_array_CC, sx_array_QC, sigma_xi_array_QC,  Sy_array_CC, sy_array_QC, sigma_yi_array_QC, Sz_array_CC, sz_array_QC, sigma_zi_array_QC = chi_square_test(N_sites, ttotal, shots)
plot_results(
    times, shots,
    sigma_x_values, sigma_y_values, sigma_z_values,
    x_values, px_values,
    ρₑₑ_array, ρ_μμ_array, ρₑμ_array,
    ρₑₑ_direct_array, ρ_μμ_direct_array, ρₑμ_direct_array,
    N_sites, ttotal, τ, trotter_steps, trotter_order,
    backend=backend,
    backend_name=backend_name,
    analytic_growth_rate=analytic_growth_rate,
    fit_t1=t1,
    fit_t2=t2,
    analytic_offset=100.0,
)
circuit.draw(output="mpl") #circuit for first timestep

