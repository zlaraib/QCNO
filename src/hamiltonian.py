#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli

from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from geometric_func import geometric_func

    
def construct_hamiltonian(N, omega, B, N_sites, Δx, delta_m_squared, p, x, Δp, theta_nu, shape_name,L, τ, energy_sign):
    H = np.zeros((2**N_sites, 2**N_sites), dtype=complex)
    # extract output of p_hat and p_mod for the p vector defined above for all sites. 
    p_mod, p_hat = momentum(p,N_sites)  
    # Construct the self-interaction Hamiltonian
    for i in range(N_sites-1):
        for j in range(i + 1, N_sites):
            geometric_factor = geometric_func(p, p_hat, i, j, theta_nu)
            interaction_strength = (np.sqrt(2) * G_F * (N[i]+ N[j])/(2*((Δx)**3)) * (1/2)) * geometric_factor
            XX = Pauli(f'{"I"*i}X{"I"*(j-i-1)}X{"I"*(N_sites-j-1)}').to_matrix()
            YY = Pauli(f'{"I"*i}Y{"I"*(j-i-1)}Y{"I"*(N_sites-j-1)}').to_matrix()
            ZZ = Pauli(f'{"I"*i}Z{"I"*(j-i-1)}Z{"I"*(N_sites-j-1)}').to_matrix()
            H += interaction_strength * (XX + YY + ZZ) 
    
            # Add Vacuum Oscillation Hamiltonian 
            if omega[i] != 0 or omega[j] != 0:
                Xi = (omega[i]/2) * B[0] * Pauli(f'{"I"*i}X{"I"*(N_sites-i-1)}').to_matrix()
                Yi = (omega[i]/2) * B[1] * Pauli(f'{"I"*i}Y{"I"*(N_sites-i-1)}').to_matrix()
                Zi = (omega[i]/2) * B[2] * Pauli(f'{"I"*i}Z{"I"*(N_sites-i-1)}').to_matrix()
                H += (1/(N_sites-1)) * (Xi + Yi + Zi)
                
                Xj = (omega[j]/2) * B[0] * Pauli(f'{"I"*j}X{"I"*(N_sites-j-1)}').to_matrix()
                Yj = (omega[j]/2) * B[1] * Pauli(f'{"I"*j}Y{"I"*(N_sites-j-1)}').to_matrix()
                Zj = (omega[j]/2) * B[2] * Pauli(f'{"I"*j}Z{"I"*(N_sites-j-1)}').to_matrix()
                H += (1/(N_sites-1)) * (Xj + Yj + Zj)

    return H