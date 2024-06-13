# evolve.py 

import numpy as np
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB

def evolve_and_measure_circuit(time, H, N_sites):
    U = Operator(expm(-1j * H * time))
    qc = QuantumCircuit(N_sites, 1)
    half_N_sites = N_sites // 2
    qc.x(range(half_N_sites))
    qc.unitary(U, range(N_sites), label="exp(-iHt)")
    qc.measure(0, 0)
    return qc

# def store_vals()
#     # Create empty arrays to store the respective values
#     Sz_array = []
#     Sy_array = []
#     Sx_array = []
#     t_array = []
#     prob_surv_array = []
#     x_values = []
#     px_values = []
#     ρₑₑ_array = []
#     ρ_μμ_array = []
#     ρₑμ_array = []

#     # Initialize variables to store ρₑμ at t1 and t2
#     ρₑμ_at_t1 = None
#     ρₑμ_at_t2 = None

#     # Example values for t1 and t2 (you can adjust these as needed)
#     t1 = 0.0
#     t2 = 1.0
#     Δt = t2 - t1  # Time difference between growth rates

#     # Extract output of p_hat and p_mod for the p vector defined above for all sites
#     p_mod, p_hat = momentum(p, N_sites)

#     # Extract px values from p_hat
#     px_values = [sub_array[0] for sub_array in p_hat]
#     return Sz_array, Sy_array, Sx_array, prob_surv_array, x_values, px_values, ρₑₑ_array, ρ_μμ_array, ρₑμ_array

