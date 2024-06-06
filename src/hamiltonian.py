#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli

def construct_hamiltonian(N_sites, interaction_strength, omega, B):
    H = np.zeros((2**N_sites, 2**N_sites), dtype=complex)
    
    for i in range(N_sites-1):
        for j in range(i + 1, N_sites):
            
            # Construct the self-interaction Hamiltonian
            XX = Pauli(f'{"I"*i}X{"I"*(j-i-1)}X{"I"*(N_sites-j-1)}').to_matrix()
            YY = Pauli(f'{"I"*i}Y{"I"*(j-i-1)}Y{"I"*(N_sites-j-1)}').to_matrix()
            ZZ = Pauli(f'{"I"*i}Z{"I"*(j-i-1)}Z{"I"*(N_sites-j-1)}').to_matrix()
            H += (interaction_strength * (XX + YY + ZZ))
            
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
    
    
# def construct_hamiltonian(N_sites, interaction_strength, omega, B):
#     H = np.zeros((2**N_sites, 2**N_sites), dtype=complex)

#     # Construct the self-interaction Hamiltonian
#     for i in range(N_sites-1):
#         for j in range(i + 1, N_sites):
#             XX = Pauli(f'{"I"*i}X{"I"*(j-i-1)}X{"I"*(N_sites-j-1)}').to_matrix()
#             YY = Pauli(f'{"I"*i}Y{"I"*(j-i-1)}Y{"I"*(N_sites-j-1)}').to_matrix()
#             ZZ = Pauli(f'{"I"*i}Z{"I"*(j-i-1)}Z{"I"*(N_sites-j-1)}').to_matrix()
#             H += interaction_strength * (XX + YY + ZZ)
    
#     # Construct the vacuum oscillation Hamiltonian
#     for k in range(N_sites):
#         if omega[k] != 0:
#             Xk = (omega[k] / 2) * B[0] * Pauli(f'{"I"*k}X{"I"*(N_sites-k-1)}').to_matrix()
#             Yk = (omega[k] / 2) * B[1] * Pauli(f'{"I"*k}Y{"I"*(N_sites-k-1)}').to_matrix()
#             Zk = (omega[k] / 2) * B[2] * Pauli(f'{"I"*k}Z{"I"*(N_sites-k-1)}').to_matrix()
#             H += (Xk + Yk + Zk)

#     return H