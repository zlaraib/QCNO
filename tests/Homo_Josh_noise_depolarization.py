#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Limit the number of cores
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["JULIA_NUM_THREADS"] = "10"

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
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
import sys
import os

from itertools import product
from qiskit.quantum_info import Pauli


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
from evolve import apply_one_timestep_dynamic_positions, apply_qubit_permutation
from perturb import pert_circuit
from activate_backend import activate_backends, build_backend


# In[ ]:


backend_name = "aer" #manila,guadalupe, aer, ibm, ionq
trotter_order= "second" #first, second
ibm_service, ionq_provider = activate_backends()


# In[ ]:


def initialize_parameters(ibm_service, ionq_provider):
    N_sites = 4 # total particles/sites/qubits for all neutrino and anti neutrino electron flavored 
    m1 = 0*eV #ergs #1st mass eigenstate of neutrino in Richers(2021)
    m2 =0*eV   #ergs #2nd mass eigenstate of neutrino in Richers(2021)
    delta_m_squared = (m2**2 - m1**2) # mass square difference # (erg^2)
 
    shots = 1042
    trotter_steps= 2 # Number of Trotter steps per time step
    optimization_level = 1
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

    # Realistic (IBM-like) depolarizing probabilities:
    p_gate1 = 0  # 0.1% depolarizing noise on single-qubit gates
    p_gate2 = 0   # 1% depolarizing noise on two-qubit gates

    error_gate1 = depolarizing_error(p_gate1, 1) # single qubit gate error actingon 1 qubit simultaneously.
    error_gate2 = depolarizing_error(p_gate2, 2) #two-qubit gate error acting on 2 qubits simulatenously, at once.

    noise_depol = NoiseModel()
    # Use the basis gates your circuit uses!
    noise_depol.add_all_qubit_quantum_error(error_gate1, ["x", "sx", "rz", "id"])  # Modern IBM basis gates
    noise_depol.add_all_qubit_quantum_error(error_gate2, ["cx"])

    print(noise_depol)

    # method and noise_model go through backend_options and are applied via
    # set_options inside build_backend.
    backend_options = dict(
        method="automatic",
        noise_model=noise_depol,
    )
    backend, euler_basis, basis_gates, two_qubit_gate = build_backend(
        backend_name, ibm_service, ionq_provider, backend_options=backend_options,
        ibm_backend="least_busy",
    )
    
    df = 1 # degrees of freedom for chisquare test
    τ =0.2*hbar  # Time step
    ttotal =100*τ  # total time of evolution in seconds # variable
    # times = np.arange(0, τ  + ttotal, τ)  # Define time steps
    times = np.arange(0, ttotal, τ)  # Define time steps
    
    L = 2.1e-11 # cm # domain size # (aka big box length)
    n_νₑ =  6.6e32 # cm^-3 # number density of electron flavor neutrino
    n_νₑ̄ =  n_νₑ # cm^-3 # number density of electron flavor antineutrino
    Eνₑ =  10*MeV # energy of all neutrinos (P.S the its negative is energy of all antineutrinos)
    Eνₑ̄ = 1 * Eνₑ # specific to my case only. Since all neutrinos have same energy, except in my case anti neutrinos are moving in opposite direction to give it a negative sign
    Δx = L  # length of the box of interacting neutrinos at a site in cm
    Δp = L # width of shape function # not being used in this test but defined to keep the evolve function arguments consistent.
    t1 = 0.008 #choose initial time for growth rate calculation #variable, not being used in this test
    t2 = 0.012  #choose final time for growth rate calculation #variable, not being used in this test
    theta_nu = 1000  #1e-6 degrees # mixing_angle # = 1.74532925E-8 radians 
    periodic = True

    mu = 1
    # Create an array of dimension N and fill it with the value 1/(sqrt(2) * G_F). This is the number of neutrinos.
    N = mu * np.full(N_sites, (Δx**3) / (np.sqrt(2) * G_F *N_sites))
    # print("N =", N)
    # Create a B vector which would be same for all N particles
    B = np.array([-np.sin(2 * theta_nu), 0, np.cos(2 * theta_nu)])
    B = B / np.linalg.norm(B)

    # Function to generate x_array such that the first particle is at position L/(2*N_sites)
    # while subsequent particles are at a position incremental by L/N_sites (grid style)
    # def generate_x_array(N_sites, L):
    #     return [(i - 0.5) * L / N_sites for i in range(1, N_sites + 1)]
    # # Generate position arrays
    # x = generate_x_array(N_sites, L)
    # y = generate_x_array(N_sites, L)
    # z = generate_x_array(N_sites, L)
    
    N_sites_eachflavor = N_sites//2
    def generate_x_array(N_sites_eachflavor, L):
        one_flavor = [
            (i - 0.5) * L / N_sites_eachflavor
            for i in range(1, N_sites_eachflavor + 1)
        ]
        return np.array(one_flavor + one_flavor)
    # Generate position arrays
    x = generate_x_array(N_sites_eachflavor, L)
    y = generate_x_array(N_sites_eachflavor, L)
    z = generate_x_array(N_sites_eachflavor, L)
    
    v_list = []
    vz_list = []
    phi_list = []
    N_draws = N_sites
    np.random.seed(42)  # Set fixed random seed for reproducibility
    for _ in range(N_draws):
        # Draw vz from a truncated Gaussian-like distribution proportional to exp(-2*(vz-1)^2) #Rejection Sampling for vz
        accepted = False  #accepted is a boolean flag that keeps trying new samples until a vz is drawn according to the desired probability distribution.
        while not accepted:
            vz = np.random.uniform(-1, 1)  # Random vz between -1 and 1
            p = np.exp(-2 * (vz - 1)**2) # Gaussian distribution for probabilities , peaks at vz=1
            if np.random.uniform(0, 1) < p:  # Accept vz with probability p
                accepted = True
        vz_list.append(vz)
        # phi uniform in [0, 2pi]
        phi = np.random.uniform(0, 2*np.pi)
        phi_list.append(phi)
        # Compute vx, vy
        vx = np.sqrt(max(0, 1 - vz**2)) * np.cos(phi)
        vy = np.sqrt(max(0, 1 - vz**2)) * np.sin(phi)
        v_list.append([vx, vy, vz])

    v = np.array(v_list)  # shape (N_sites, 3)
    for i in range(N_sites):
        v[i, :] /= np.linalg.norm(v[i, :]) 
    print("Velocity vector v:\n", v)
    p=v
    omega = np.zeros(N_sites)
    print(omega)
    #modify this shape_name dict thing
    shape_name = "none"  # variable.  #Select a shape function based on the shape_name variable form the list defined in dictionary in shape_func file

    B_pert= None
    alpha=0
    geometric_name= "physical"
    energy_sign = np.ones(N_sites)
    bit_list = ['0'] * (N_sites//2 + 1) + ['1'] * (N_sites - N_sites//2 - 1)
    advection = False #True/False
    return N_sites, theta_nu, geometric_name,bit_list,omega, B, B_pert, alpha, N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ, shape_name , shots, trotter_steps, backend,backend_name, df, τ, times,ttotal,optimization_level, periodic,two_qubit_gate, euler_basis, basis_gates,advection,p_gate1, p_gate2


# In[ ]:





# In[ ]:


def get_counts_and_sigmas(qc_base, N_sites, backend,backend_name, optimization_level, shots, two_qubit_gate, euler_basis, basis_gates,df=0):
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

    return {
        "counts_z": counts_z,
        "counts_x": counts_x,
        "counts_y": counts_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
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


def stabilizer_renyi_magic_from_rdm(rdm, eps=1e-15, return_pauli_weights=False):

    rho = np.asarray(rdm.data, dtype=complex)
    dim = rho.shape[0]
    n_qubits = int(np.round(np.log2(dim)))

    if 2 ** n_qubits != dim:
        raise ValueError(f"RDM dimension {dim} is not a power of 2.")

    pauli_labels = ["I", "X", "Y", "Z"]

    sum_cP4 = 0.0
    pauli_weight_distribution = np.zeros(n_qubits + 1, dtype=float)

    for label_tuple in product(pauli_labels, repeat=n_qubits):
        label = "".join(label_tuple)

        weight = sum(ch != "I" for ch in label)

        P = Pauli(label).to_matrix()
        cP = np.trace(rho @ P)

        abs_cP2 = np.abs(cP) ** 2
        abs_cP4 = np.abs(cP) ** 4

        sum_cP4 += abs_cP4
        pauli_weight_distribution[weight] += abs_cP2

    magic_value = sum_cP4 / dim
    magic_value = max(np.real_if_close(magic_value).item(), eps)
    magic = -np.log2(magic_value)

    pauli_weight_distribution = pauli_weight_distribution / dim

    if return_pauli_weights:
        return float(magic), pauli_weight_distribution

    return float(magic)


# In[ ]:


def evolve_and_simulate(
    times, omega, energy_sign, delta_m_squared, Eνₑ, t1, t2,
    N, B, B_pert, alpha, Δx, Δp, shape_name, N_sites, x, p, L,
    theta_nu, geometric_name, bit_list, backend, backend_name, shots, τ, df, ttotal,
    optimization_level, periodic, trotter_steps, trotter_order,
    two_qubit_gate, euler_basis, basis_gates,advection, p_gate1, p_gate2
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

    x_values = []
    px_values = []
    py_values = []
    pz_values = []
    
    
    sigma_x_direct_array = []
    sigma_y_direct_array = []
    sigma_z_direct_array = []
    
    renyi2_single_array = []
    renyi2_two_site_array = []
    renyi2_bipartite_array = []

    magic_single_array = []
    magic_two_site_array = []
    magic_bipartite_array = []
    magic_global_array = []

    pauli_weight_array = []
    density_matrix_data_array = []
    
    two_site_pairs = [(i, j) for i in range(N_sites) for j in range(i + 1, N_sites)]

    def reorder_sorted_to_original(data_sorted, particle_ids):
        data_sorted = np.asarray(data_sorted)
        data_original = np.empty_like(data_sorted)
        for sorted_slot, pid in enumerate(particle_ids):
            data_original[pid] = data_sorted[sorted_slot]
        return data_original
 
    def write_row(fhandle, t, values):
        row = np.concatenate(([t], np.asarray(values, dtype=float).ravel()))
        np.savetxt(fhandle, row[None, :], fmt="%.16e")
    
    def write_scalar_row(fhandle, t, value):
        np.savetxt(fhandle, np.array([[t, float(value)]]), fmt="%.16e")


    def renyi2_from_rdm(rdm, eps=1e-15):
        rho = np.asarray(rdm.data, dtype=complex)
        purity = np.real(np.trace(rho @ rho))
        purity = np.clip(purity, eps, 1.0)
        return -np.log2(purity)

    np.savetxt(
        os.path.join(datadir, "two_site_pairs.dat"),
        np.array(two_site_pairs, dtype=int),
        fmt="%d"
    )

    np.savetxt(
        os.path.join(datadir, "pauli_weights.dat"),
        np.arange(N_sites + 1, dtype=int),
        fmt="%d"
    )
    
    # original physical labels
    particle_ids = np.arange(N_sites)

    # initial sort
    perm0 = np.argsort(x, kind="stable")
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
        "py": open(os.path.join(datadir, "t_pysiteval.dat"), "w"),
        "pz": open(os.path.join(datadir, "t_pzsiteval.dat"), "w"),
        "sx": open(os.path.join(datadir, "t_sigma_x.dat"), "w"),
        "sy": open(os.path.join(datadir, "t_sigma_y.dat"), "w"),
        "sz": open(os.path.join(datadir, "t_sigma_z.dat"), "w"),
        "sigma_x_direct": open(os.path.join(datadir, "t_sigma_x_direct.dat"), "w"),
        "sigma_y_direct": open(os.path.join(datadir, "t_sigma_y_direct.dat"), "w"),
        "sigma_z_direct": open(os.path.join(datadir, "t_sigma_z_direct.dat"), "w"),
        
        "renyi2_single": open(os.path.join(datadir, "t_renyi2_single_direct.dat"), "w"),
        "renyi2_two_site": open(os.path.join(datadir, "t_renyi2_two_site_direct.dat"), "w"),
        "renyi2_bipartite": open(os.path.join(datadir, "t_renyi2_bipartite_direct.dat"), "w"),

        "magic_single": open(os.path.join(datadir, "t_magic_single_direct.dat"), "w"),
        "magic_two_site": open(os.path.join(datadir, "t_magic_two_site_direct.dat"), "w"),
        "magic_bipartite": open(os.path.join(datadir, "t_magic_bipartite_direct.dat"), "w"),
        "magic_global": open(os.path.join(datadir, "t_magic_global_direct.dat"), "w"),

        "pauli_weight": open(os.path.join(datadir, "t_pauli_weight_distribution_direct.dat"), "w"),
    }

    try:
        for step_idx, t in enumerate(times):
            print(f"\niteration={step_idx}, time={t}")

            # -------------------------------
            # Store x and px in original order
            # -------------------------------
            x_original = reorder_sorted_to_original(x, particle_ids)
            px_original = reorder_sorted_to_original(p[:, 0], particle_ids)
            py_original = reorder_sorted_to_original(p[:, 1], particle_ids)
            pz_original = reorder_sorted_to_original(p[:, 2], particle_ids)

            x_values.append(x_original.copy())
            px_values.append(px_original.copy())
            py_values.append(py_original.copy())
            pz_values.append(pz_original.copy())

            write_row(file_handles["x"], t, x_original)
            write_row(file_handles["px"], t, px_original)
            write_row(file_handles["py"], t, py_original)
            write_row(file_handles["pz"], t, pz_original)

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

            print("counts_z =", meas_data["counts_z"])

            # -------------------------------------------------
            # Keep current sorted-site order, matching Julia.
            # Do NOT reorder back to original particle order.
            # -------------------------------------------------
            sigma_x_current = sigma_x_sorted.copy()
            sigma_y_current = sigma_y_sorted.copy()
            sigma_z_current = sigma_z_sorted.copy()

            sigma_x_values.append(sigma_x_current.copy())
            sigma_y_values.append(sigma_y_current.copy())
            sigma_z_values.append(sigma_z_current.copy())

            write_row(file_handles["sx"], t, sigma_x_current)
            write_row(file_handles["sy"], t, sigma_y_current)
            write_row(file_handles["sz"], t, sigma_z_current)

            print(f"t={t} sigma_x(sorted site order) = {sigma_x_current}")
            print(f"t={t} sigma_y(sorted site order) = {sigma_y_current}")
            print(f"t={t} sigma_z(sorted site order) = {sigma_z_current}")
            
            if direct_ok:
                _, density_matrix_z, statevector_z = get_direct_state(
                    qc_base, backend, optimization_level
                )

                sigma_x_direct_sites = []
                sigma_y_direct_sites = []
                sigma_z_direct_sites = []
                
                
                density_matrix_data_array.append(
                    np.asarray(density_matrix_z.data, dtype=complex).copy()
                )


                # ---------------------------------------------------------
                # Helper for partial_trace with the same source/Juliastyle
                # site convention used for sigma components.
                #
                # site index = Pauli-label position
                # Qiskit qubit index = N_sites - 1 - site
                # ---------------------------------------------------------
                def traced_out_for_sites(keep_sites):
                    keep_qubits = {
                        N_sites - 1 - int(site)
                        for site in keep_sites
                    }
                    return [
                        q for q in range(N_sites)
                        if q not in keep_qubits
                    ]

                for site in range(N_sites):
                    # ---------------------------------------------------------
                    # Source/Juliastyle site convention:
                    #
                    # Pauli label position "site" corresponds to source site.
                    # Qiskit internally maps label position site -> qubit N-1-site.
                    #
                    # Therefore this matches the same convention as construct_hamiltonian().
                    # Direct sigma values
                    # ---------------------------------------------------------

                    label_x = ["I"] * N_sites
                    label_y = ["I"] * N_sites
                    label_z = ["I"] * N_sites

                    label_x[site] = "X"
                    label_y[site] = "Y"
                    label_z[site] = "Z"

                    op_x = Pauli("".join(label_x))
                    op_y = Pauli("".join(label_y))
                    op_z = Pauli("".join(label_z))

                    sx = statevector_z.expectation_value(op_x)
                    sy = statevector_z.expectation_value(op_y)
                    sz = statevector_z.expectation_value(op_z)

                    sigma_x_direct_sites.append(np.real_if_close(sx).real)
                    sigma_y_direct_sites.append(np.real_if_close(sy).real)
                    sigma_z_direct_sites.append(np.real_if_close(sz).real)

                sigma_x_direct_sites = np.asarray(sigma_x_direct_sites, dtype=float)
                sigma_y_direct_sites = np.asarray(sigma_y_direct_sites, dtype=float)
                sigma_z_direct_sites = np.asarray(sigma_z_direct_sites, dtype=float)
            
                # -------------------------------
                # Single-site quantities
                # Keep sorted-site order.
                # Do NOT reorder back to original particle order.
                # -------------------------------
                
                renyi2_single_sites = []
                magic_single_sites = []

                for site in range(N_sites):
                    traced_out = traced_out_for_sites([site])
                    rdm_1 = partial_trace(density_matrix_z, traced_out)

                    renyi2_single_sites.append(renyi2_from_rdm(rdm_1))
                    magic_single_sites.append(stabilizer_renyi_magic_from_rdm(rdm_1))

                renyi2_single_current = np.asarray(renyi2_single_sites, dtype=float)
                magic_single_current = np.asarray(magic_single_sites, dtype=float)

                renyi2_single_array.append(renyi2_single_current.copy())
                magic_single_array.append(magic_single_current.copy())

                write_row(file_handles["renyi2_single"], t, renyi2_single_current)
                write_row(file_handles["magic_single"], t, magic_single_current)

                # -------------------------------
                # Two-site quantities
                # Keep sorted-site pair order.
                # Pair order matches two_site_pairs.dat.
                # Do NOT use particle_ids here.
                # -------------------------------
                
                renyi2_two_site_values = []
                magic_two_site_values = []

                for site_i, site_j in two_site_pairs:
                    traced_out = traced_out_for_sites([site_i, site_j])
                    rdm_2 = partial_trace(density_matrix_z, traced_out)

                    renyi2_two_site_values.append(renyi2_from_rdm(rdm_2))
                    magic_two_site_values.append(stabilizer_renyi_magic_from_rdm(rdm_2))

                renyi2_two_site_vec = np.asarray(renyi2_two_site_values, dtype=float)
                magic_two_site_vec = np.asarray(magic_two_site_values, dtype=float)

                renyi2_two_site_array.append(renyi2_two_site_vec.copy())
                magic_two_site_array.append(magic_two_site_vec.copy())

                write_row(file_handles["renyi2_two_site"], t, renyi2_two_site_vec)
                write_row(file_handles["magic_two_site"], t, magic_two_site_vec)

                # -------------------------------
                # Bipartite quantities
                # -------------------------------
                half = N_sites // 2
                subsystem_A_current = list(range(half))

                traced_out_bip = traced_out_for_sites(subsystem_A_current)

                rdm_bip = partial_trace(density_matrix_z, traced_out_bip)

                renyi2_bip = renyi2_from_rdm(rdm_bip)
                magic_bip = stabilizer_renyi_magic_from_rdm(rdm_bip)

                # -------------------------------
                # Global magic + Pauli weights
                # No site reordering needed.
                # Pauli weights are global weight sectors, not site-indexed.
                # -------------------------------
                magic_global, pauli_weights = stabilizer_renyi_magic_from_rdm(
                    density_matrix_z,
                    return_pauli_weights=True
                )

                renyi2_bipartite_array.append(renyi2_bip)
                magic_bipartite_array.append(magic_bip)
                magic_global_array.append(magic_global)
                pauli_weight_array.append(pauli_weights.copy())

                # -------------------------------
                # Direct sigma arrays
                # Keep sorted-site order, matching sigma components above.
                # -------------------------------
                sigma_x_direct_current = sigma_x_direct_sites.copy()
                sigma_y_direct_current = sigma_y_direct_sites.copy()
                sigma_z_direct_current = sigma_z_direct_sites.copy()

                sigma_x_direct_array.append(sigma_x_direct_current.copy())
                sigma_y_direct_array.append(sigma_y_direct_current.copy())
                sigma_z_direct_array.append(sigma_z_direct_current.copy())

                write_row(file_handles["sigma_x_direct"], t, sigma_x_direct_current)
                write_row(file_handles["sigma_y_direct"], t, sigma_y_direct_current)
                write_row(file_handles["sigma_z_direct"], t, sigma_z_direct_current)

                write_scalar_row(file_handles["renyi2_bipartite"], t, renyi2_bip)
                write_scalar_row(file_handles["magic_bipartite"], t, magic_bip)
                write_scalar_row(file_handles["magic_global"], t, magic_global)
                write_row(file_handles["pauli_weight"], t, pauli_weights)

                print(f"t={t} single-site Renyi-2(sorted site order) = {renyi2_single_current}")
                print(f"t={t} two-site Renyi-2(sorted pair order)    = {renyi2_two_site_vec}")
                print(f"t={t} bipartite Renyi-2(sorted sites)       = {renyi2_bip}")
                print(f"t={t} single-site Magic(sorted site order)   = {magic_single_current}")
                print(f"t={t} two-site Magic(sorted pair order)      = {magic_two_site_vec}")
                print(f"t={t} bipartite Magic(sorted sites)          = {magic_bip}")
                print(f"t={t} global Magic                           = {magic_global}")
                print(f"t={t} Pauli weights                          = {pauli_weights}")

                print(f"t={t} sigma_x direct(source/Juliastyle site order) = {sigma_x_direct_current}")
                print(f"t={t} sigma_y direct(source/Juliastyle site order) = {sigma_y_direct_current}")
                print(f"t={t} sigma_z direct(source/Juliastyle site order) = {sigma_z_direct_current}")
            else:
                nan_vec = np.full(N_sites, np.nan, dtype=float)
                nan_pairs = np.full(len(two_site_pairs), np.nan, dtype=float)
                nan_weights = np.full(N_sites + 1, np.nan, dtype=float)

                # Be careful: this is 2^N x 2^N, but okay for small N.
                nan_dm = np.full((2**N_sites, 2**N_sites), np.nan, dtype=complex)

                density_matrix_data_array.append(nan_dm)

                # Direct sigma placeholders
                sigma_x_direct_array.append(nan_vec.copy())
                sigma_y_direct_array.append(nan_vec.copy())
                sigma_z_direct_array.append(nan_vec.copy())

                write_row(file_handles["sigma_x_direct"], t, nan_vec)
                write_row(file_handles["sigma_y_direct"], t, nan_vec)
                write_row(file_handles["sigma_z_direct"], t, nan_vec)

                # Renyi entropy placeholders
                renyi2_single_array.append(nan_vec.copy())
                renyi2_two_site_array.append(nan_pairs.copy())
                renyi2_bipartite_array.append(np.nan)

                write_row(file_handles["renyi2_single"], t, nan_vec)
                write_row(file_handles["renyi2_two_site"], t, nan_pairs)
                write_scalar_row(file_handles["renyi2_bipartite"], t, np.nan)

                # Magic placeholders
                magic_single_array.append(nan_vec.copy())
                magic_two_site_array.append(nan_pairs.copy())
                magic_bipartite_array.append(np.nan)
                magic_global_array.append(np.nan)

                write_row(file_handles["magic_single"], t, nan_vec)
                write_row(file_handles["magic_two_site"], t, nan_pairs)
                write_scalar_row(file_handles["magic_bipartite"], t, np.nan)
                write_scalar_row(file_handles["magic_global"], t, np.nan)

                # Pauli-weight placeholders
                pauli_weight_array.append(nan_weights.copy())
                write_row(file_handles["pauli_weight"], t, nan_weights)            
            
            
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

    # -------------------------------
    # Convert lists to arrays
    # -------------------------------
    times = np.asarray(times, dtype=float)

    x_values = np.asarray(x_values, dtype=float)
    px_values = np.asarray(px_values, dtype=float)
    py_values = np.asarray(py_values, dtype=float)
    pz_values = np.asarray(pz_values, dtype=float)

    sigma_x_values = np.asarray(sigma_x_values, dtype=float)
    sigma_y_values = np.asarray(sigma_y_values, dtype=float)
    sigma_z_values = np.asarray(sigma_z_values, dtype=float)

    sigma_x_direct_array = np.asarray(sigma_x_direct_array, dtype=float)
    sigma_y_direct_array = np.asarray(sigma_y_direct_array, dtype=float)
    sigma_z_direct_array = np.asarray(sigma_z_direct_array, dtype=float)

    renyi2_single_array = np.asarray(renyi2_single_array, dtype=float)
    renyi2_two_site_array = np.asarray(renyi2_two_site_array, dtype=float)
    renyi2_bipartite_array = np.asarray(renyi2_bipartite_array, dtype=float)

    magic_single_array = np.asarray(magic_single_array, dtype=float)
    magic_two_site_array = np.asarray(magic_two_site_array, dtype=float)
    magic_bipartite_array = np.asarray(magic_bipartite_array, dtype=float)
    magic_global_array = np.asarray(magic_global_array, dtype=float)

    pauli_weight_array = np.asarray(pauli_weight_array, dtype=float)
    density_matrix_data_array = np.asarray(density_matrix_data_array, dtype=complex)

    # -------------------------------
    # Save density matrices and global metadata
    # -------------------------------
    np.savez_compressed(
        os.path.join(datadir, "density_matrices_direct.npz"),
        times=times,
        density_matrices=density_matrix_data_array,
        pauli_weights=pauli_weight_array,
        two_site_pairs=np.asarray(two_site_pairs, dtype=int),
        p1_gate=globals().get("p_gate1", np.nan),
        p2_gate=globals().get("p_gate2", np.nan),
    )

    return (
        sigma_x_values,
        sigma_y_values,
        sigma_z_values,

        sigma_x_direct_array,
        sigma_y_direct_array,
        sigma_z_direct_array,

        x_values,
        px_values,
        py_values,
        pz_values,

        renyi2_single_array,
        renyi2_two_site_array,
        renyi2_bipartite_array,

        magic_single_array,
        magic_two_site_array,
        magic_bipartite_array,
        magic_global_array,

        pauli_weight_array,
        density_matrix_data_array,
        two_site_pairs,

        isa_circuit_z_first,
    )


# In[ ]:


def plot_results(
    times, shots,
    sigma_x_values, sigma_y_values, sigma_z_values,
    sigma_x_direct_array,
    sigma_y_direct_array,
    sigma_z_direct_array,
    x_values, px_values,
    renyi2_single_array, renyi2_two_site_array, renyi2_bipartite_array,
    magic_single_array, magic_two_site_array, magic_bipartite_array, magic_global_array,
    pauli_weight_array,
    two_site_pairs,
    N_sites, ttotal, τ, trotter_steps, trotter_order
):
    plotdir = os.path.join(
        os.getcwd(), "plots"
    )
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    times = np.asarray(times, dtype=float) /hbar
    sigma_x_values = np.asarray(sigma_x_values, dtype=float)
    sigma_y_values = np.asarray(sigma_y_values, dtype=float)
    sigma_z_values = np.asarray(sigma_z_values, dtype=float)

    x_values = np.asarray(x_values, dtype=float)
    px_values = np.asarray(px_values, dtype=float)

    renyi2_single_array = np.asarray(renyi2_single_array, dtype=float)
    renyi2_two_site_array = np.asarray(renyi2_two_site_array, dtype=float)
    renyi2_bipartite_array = np.asarray(renyi2_bipartite_array, dtype=float)

    magic_single_array = np.asarray(magic_single_array, dtype=float)
    magic_two_site_array = np.asarray(magic_two_site_array, dtype=float)
    magic_bipartite_array = np.asarray(magic_bipartite_array, dtype=float)
    magic_global_array = np.asarray(magic_global_array, dtype=float)

    pauli_weight_array = np.asarray(pauli_weight_array, dtype=float)
    
    def plot_all_sites(data, ylabel, title, filename, semilogy=False):
        plt.figure()
        for site in range(N_sites):
            y = data[:, site]
            if semilogy:
                # y = np.maximum(np.abs(y), 1e-300)
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
        
    # =========================
    # Sigma plots from counts
    # =========================
    plot_all_sites(
        sigma_x_values,
        ylabel=r"$\sigma_x$",
        title=fr"$\sigma_x$ vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_x_vs_time.pdf",
        semilogy=False
    )

    plot_all_sites(
        sigma_y_values,
        ylabel=r"$\sigma_y$",
        title=fr"$\sigma_y$ vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_y_vs_time.pdf",
        semilogy=False
    )

    plot_all_sites(
        sigma_z_values,
        ylabel=r"$\sigma_z$",
        title=fr"$\sigma_z$ vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_z_vs_time.pdf",
        semilogy=False
    )
    plot_all_sites(
        sigma_x_direct_array,
        ylabel=r"$\sigma_x$ (direct state)",
        title=fr"$\sigma_x$ (direct state) vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_x_direct_vs_time.pdf",
        semilogy=False
    )  
    plot_all_sites(
        sigma_y_direct_array,
        ylabel=r"$\sigma_y$ (direct state)",
        title=fr"$\sigma_y$ (direct state) vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_y_direct_vs_time.pdf",
        semilogy=False
    )        
    plot_all_sites(
        sigma_z_direct_array,
        ylabel=r"$\sigma_z$ (direct state)",
        title=fr"$\sigma_z$ (direct state) vs Time for all sites ({N_sites} sites)",
        filename="Josh_sigma_z_direct_vs_time.pdf",
        semilogy=False
    )   

       
    # =========================
    # Single-site Rényi-2
    # =========================
    plot_all_sites(
        renyi2_single_array,
        ylabel=r"$S_2$",
        title=fr"Single-site 2nd R\'enyi entropy vs Time ({N_sites} sites)",
        filename="Josh_renyi2_single_direct_vs_time.pdf"
    )

    # =========================
    # Two-site Rényi-2
    # =========================
    plot_all_pairs(
        renyi2_two_site_array,
        two_site_pairs,
        ylabel=r"$S_2$",
        title=fr"Two-site 2nd R\'enyi entropy vs Time ({N_sites} sites)",
        filename="Josh_renyi2_two_site_direct_vs_time.pdf",
        ncol=2
    )

    # =========================
    # Bipartite Rényi-2
    # =========================
    plt.figure()
    plt.plot(times, renyi2_bipartite_array, label="bipartition")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$S_2$")
    plt.title(fr"Bipartite 2nd R\'enyi entropy vs Time ({N_sites} sites)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_renyi2_bipartite_direct_vs_time.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Single-site magic
    # =========================
    plot_all_sites(
        magic_single_array,
        ylabel=r"$M_2$",
        title=fr"Single-site stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_single_direct_vs_time.pdf"
    )

    # =========================
    # Two-site magic
    # =========================
    plot_all_pairs(
        magic_two_site_array,
        two_site_pairs,
        ylabel=r"$M_2$",
        title=fr"Two-site stabilizer R\'enyi magic vs Time ({N_sites} sites)",
        filename="Josh_magic_two_site_direct_vs_time.pdf",
        ncol=2
    )

    # =========================
    # Bipartite magic
    # =========================
    plt.figure()
    plt.plot(times, magic_bipartite_array, label="bipartition")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$M_2$")
    plt.title(fr"Bipartite stabilizer R\'enyi magic vs Time ({N_sites} sites)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_magic_bipartite_direct_vs_time.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Global magic
    # =========================
    plt.figure()
    plt.plot(times, magic_global_array, label="global")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$M_2$")
    plt.title(fr"Global stabilizer R\'enyi magic vs Time ({N_sites} sites)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_magic_global_direct_vs_time.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # Pauli weight distribution
    # =========================
    plt.figure()

    max_weight = pauli_weight_array.shape[1] - 1

    for w in range(max_weight + 1):
        plt.plot(times, pauli_weight_array[:, w], label=fr"$w={w}$")

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
        y = np.maximum(np.abs(pauli_weight_array[:, w]), 1e-300)
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
    norm = np.sum(pauli_weight_array, axis=1)
    norm = np.maximum(norm, 1e-300)

    average_pauli_weight = np.sum(pauli_weight_array * weights[None, :], axis=1) / norm

    plt.figure()
    plt.plot(times, average_pauli_weight, label=r"$\langle w \rangle$")
    plt.xlabel("Time/ ħ")
    plt.ylabel(r"$\langle w \rangle$")
    plt.title(fr"Average Pauli weight vs Time ({N_sites} qubits)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_average_pauli_weight_vs_time.pdf"), bbox_inches="tight")
    plt.show()



    # =========================
    # Position evolution
    # =========================
    plt.figure()
    plt.title(f"Position Evolution for {N_sites} sites")
    plt.ylabel("Time/ ħ")
    plt.xlabel("Position (x)")
    for site in range(N_sites):
        plt.plot( x_values[:, site], times,label=f"Site {site+1}")
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
        plt.plot(px_values[:, site],times, label=f"Site {site+1}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "Josh_Particles_momentum_evolution.pdf"), bbox_inches="tight")
    plt.show()


# In[ ]:


N_sites, theta_nu, geometric_name,bit_list,omega, B, B_pert,alpha, N, x, p, energy_sign, Δx, Δp, L, delta_m_squared, t1, t2, Eνₑ, shape_name , shots, trotter_steps, backend,backend_name, df, τ, times,ttotal,optimization_level, periodic, two_qubit_gate, euler_basis, basis_gates, advection,p_gate1, p_gate2= initialize_parameters(ibm_service, ionq_provider)
(sigma_x_values,
sigma_y_values,
sigma_z_values,

sigma_x_direct_array,
sigma_y_direct_array,
sigma_z_direct_array,

x_values,
px_values,
py_values,
pz_values,

renyi2_single_array,
renyi2_two_site_array,
renyi2_bipartite_array,

magic_single_array,
magic_two_site_array,
magic_bipartite_array,
magic_global_array,

pauli_weight_array,
density_matrix_data_array,
two_site_pairs,
circuit)= evolve_and_simulate(times, omega,energy_sign, delta_m_squared, Eνₑ, t1,t2, N, B, B_pert,alpha,Δx,Δp, shape_name, N_sites,x, p, L, theta_nu,geometric_name, bit_list,backend,backend_name, shots, τ, df, ttotal, optimization_level,periodic, trotter_steps, trotter_order,two_qubit_gate, euler_basis, basis_gates, advection,p_gate1, p_gate2)
# chi_square_value, Sx_array_CC, sx_array_QC, sigma_xi_array_QC,  Sy_array_CC, sy_array_QC, sigma_yi_array_QC, Sz_array_CC, sz_array_QC, sigma_zi_array_QC = chi_square_test(N_sites, ttotal, shots)
plot_results(times, shots,
    sigma_x_values, sigma_y_values, sigma_z_values,
    sigma_x_direct_array,
    sigma_y_direct_array,
    sigma_z_direct_array,
    x_values, px_values,
    renyi2_single_array, renyi2_two_site_array, renyi2_bipartite_array,
    magic_single_array, magic_two_site_array, magic_bipartite_array, magic_global_array,
    pauli_weight_array,
    two_site_pairs,
    N_sites, ttotal, τ, trotter_steps, trotter_order)
# circuit.draw(output="mpl")  #circuit for first timestep


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




