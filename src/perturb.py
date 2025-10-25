#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Pauli, Operator


def construct_pert_hamiltonian( omega, B_pert, N_sites):
    pauli_terms = []

    for i in range(N_sites):

                Xi = Pauli(f'{"I"*i}X{"I"*(N_sites-i-1)}')
                Yi = Pauli(f'{"I"*i}Y{"I"*(N_sites-i-1)}')
                Zi = Pauli(f'{"I"*i}Z{"I"*(N_sites-i-1)}')
                
                pauli_terms.append(( B_pert[0], Xi))
                pauli_terms.append(( B_pert[1], Yi))
                pauli_terms.append(( B_pert[2], Zi))       
    return pauli_terms

    # for i in range(N_sites - 1):
    #     for j in range(i + 1, N_sites):
    #             Xi = Pauli(f'{"I"*i}X{"I"*(N_sites-i-1)}')
    #             Yi = Pauli(f'{"I"*i}Y{"I"*(N_sites-i-1)}')
    #             Zi = Pauli(f'{"I"*i}Z{"I"*(N_sites-i-1)}')
                
    #             pauli_terms.append(((omega[i] / 2) * B_pert[0]/ (N_sites - 1), Xi))
    #             pauli_terms.append(((omega[i] / 2) * B_pert[1]/ (N_sites - 1), Yi))
    #             pauli_terms.append(((omega[i] / 2) * B_pert[2]/ (N_sites - 1), Zi))
                
    #             Xj = Pauli(f'{"I"*j}X{"I"*(N_sites-j-1)}')
    #             Yj = Pauli(f'{"I"*j}Y{"I"*(N_sites-j-1)}')
    #             Zj = Pauli(f'{"I"*j}Z{"I"*(N_sites-j-1)}')
                
    #             pauli_terms.append(((omega[j] / 2) * B_pert[0]/ (N_sites - 1), Xj))
    #             pauli_terms.append(((omega[j] / 2) * B_pert[1]/ (N_sites - 1), Yj))
    #             pauli_terms.append(((omega[j] / 2) * B_pert[2]/ (N_sites - 1), Zj))
    
    # return pauli_terms
def apply_single_qubit_gate(qc, coef, qubit, pauli):
    if pauli == 'X':
        qc.rx(2 * coef, qubit)
    elif pauli == 'Y':
        qc.ry(2 * coef, qubit)
    elif pauli == 'Z':
        qc.rz(2 * coef, qubit)
# def apply_single_qubit_gate(qc, coef, qubit, pauli):
#     if pauli == 'X':
#         qc.u(2 * coef, -np.pi / 2, np.pi / 2, qubit)  # u(theta, phi, lambda, qubit)
#     elif pauli == 'Y':
#         qc.u(2 * coef, 0, 0, qubit)
#     elif pauli == 'Z':
#         qc.u(2 * coef, 0, np.pi / 2, qubit)

# def apply_single_qubit_gate(qc, coef, qubit, pauli):

#     qc.u(2 * coef, 2 * coef,0, qubit)  # u(theta, phi, lambda, qubit)

def pert_circuit(qc, dt_substep, omega, B_pert, N_sites, measure='Z'):
    pauli_terms = construct_pert_hamiltonian( omega, B_pert, N_sites)
    # print("Pert Pauli Terms:", pauli_terms)

    for coef, pauli in pauli_terms:
        #converts the Pauli operator object into its string representation
        pauli_str = pauli.to_label()
        
        #Creates a list of qubits that are affected by the Pauli term.
        #uses a list comprehension that iterates over the string representation pauli_str of the Pauli operator.
        # enumerate(pauli_str) provides both the index i and the character p for each position in the string.
        #the condition if p != 'I' ensures that only qubits with a Pauli operator other than the identity I are included in the list qubits.
        qubits = [i for i, p in enumerate(pauli_str) if p != 'I']

        #coef * dt_substep: The rotation angle, which is the coefficient coef multiplied by the time step dt_substep
        apply_single_qubit_gate(qc, coef * dt_substep, qubits[0], pauli_str[qubits[0]])
        

    # for qubit in range(N_sites):
    #         # Randomly select an axis (X, Y, Z) and a random angle between 0 and 2*pi
    #         axis = np.random.choice(['X', 'Y', 'Z'])
    #         angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2*pi

    #         if axis == 'X':
    #             qc.rx(angle, qubit)  # Apply rotation around X-axis
    #         elif axis == 'Y':
    #             qc.ry(angle, qubit)  # Apply rotation around Y-axis
    #         elif axis == 'Z':
    #             qc.rz(angle, qubit)  # Apply rotation around Z-axis


    
    # transform basis from Z (up/down or |0>/|1>) to X (equal super position of |up>/ |down> to get the x basis =|+>/|->) using hadamard on the first qubit(b/c we are measurng the first qubit only)
    # the measurement of counts in x basis is done later in main script to compute <sigma_x>
    if measure == 'X':
        qc.h(range(N_sites))  # Apply Hadamard to all qubits for X basis measurement
        
    # prepare the quantum state by transforming basis from Z to Y using sdg on the first qubit(b/c we are measurng the first qubit only)
    # Similarly, the 'Sdg' gate (S-dagger) is applied, which corresponds to the inverse of the phase gate 'S'. This introduces a phase shift of - pi/2.
    # Then, a Hadamard gate 'H' is applied. This sequence of gates transforms from Z basis to Y basis. 
    
    # the measurement of counts in Y basis is done later in main script to compute <sigma_y>
    elif measure == 'Y':
        qc.sdg(range(N_sites))  # Apply Sdg to all qubits for Y basis measurement
        qc.h(range(N_sites))  # Then apply Hadamard   
        
    return qc


