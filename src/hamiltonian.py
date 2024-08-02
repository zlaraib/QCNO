#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Pauli, Operator
from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from geometric_func import geometric_func


def construct_hamiltonian(N, omega, B, N_sites, Δx, delta_m_squared, p, x, Δp, theta_nu, shape_name, L, τ, energy_sign):
    pauli_terms = []
    p_mod, p_hat = momentum(p, N_sites)  
    
    for i in range(N_sites - 1):
        for j in range(i + 1, N_sites):
            geometric_factor = geometric_func(p, p_hat, i, j, theta_nu)
            interaction_strength = (np.sqrt(2) * G_F * (N[i] + N[j]) / (2 * (Δx)**3) * (1 / 2)) * geometric_factor
            
            if interaction_strength != 0:
                XX = Pauli(f'{"I"*i}X{"I"*(j-i-1)}X{"I"*(N_sites-j-1)}')
                YY = Pauli(f'{"I"*i}Y{"I"*(j-i-1)}Y{"I"*(N_sites-j-1)}')
                ZZ = Pauli(f'{"I"*i}Z{"I"*(j-i-1)}Z{"I"*(N_sites-j-1)}')
                
                pauli_terms.append((interaction_strength, XX))
                pauli_terms.append((interaction_strength, YY))
                pauli_terms.append((interaction_strength, ZZ))
    
            if omega[i] != 0 or omega[j] != 0:
                Xi = Pauli(f'{"I"*i}X{"I"*(N_sites-i-1)}')
                Yi = Pauli(f'{"I"*i}Y{"I"*(N_sites-i-1)}')
                Zi = Pauli(f'{"I"*i}Z{"I"*(N_sites-i-1)}')
                
                pauli_terms.append(((omega[i] / 2) * B[0]/ (N_sites - 1), Xi))
                pauli_terms.append(((omega[i] / 2) * B[1]/ (N_sites - 1), Yi))
                pauli_terms.append(((omega[i] / 2) * B[2]/ (N_sites - 1), Zi))
                
                Xj = Pauli(f'{"I"*j}X{"I"*(N_sites-j-1)}')
                Yj = Pauli(f'{"I"*j}Y{"I"*(N_sites-j-1)}')
                Zj = Pauli(f'{"I"*j}Z{"I"*(N_sites-j-1)}')
                
                pauli_terms.append(((omega[j] / 2) * B[0]/ (N_sites - 1), Xj))
                pauli_terms.append(((omega[j] / 2) * B[1]/ (N_sites - 1), Yj))
                pauli_terms.append(((omega[j] / 2) * B[2]/ (N_sites - 1), Zj))
    
    return pauli_terms