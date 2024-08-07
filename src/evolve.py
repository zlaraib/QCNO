# evolve.py 

import numpy as np
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit


from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp, Operator

from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from geometric_func import geometric_func
from hamiltonian import construct_hamiltonian

def apply_single_qubit_gate(qc, coef, qubit, pauli):
    if pauli == 'X':
        qc.rx(2 * coef, qubit)
    elif pauli == 'Y':
        qc.ry(2 * coef, qubit)
    elif pauli == 'Z':
        qc.rz(2 * coef, qubit)

def apply_two_qubit_gate(qc, coef, qubit1, qubit2, pauli1, pauli2):
    if pauli1 == 'X' and pauli2 == 'X':
        qc.rxx(2 * coef, qubit1, qubit2)
    elif pauli1 == 'Y' and pauli2 == 'Y':
        qc.ryy(2 * coef, qubit1, qubit2)
    elif pauli1 == 'Z' and pauli2 == 'Z':
        qc.rzz(2 * coef, qubit1, qubit2)

def evolve_and_measure_circuit(time, pauli_terms,  N_sites, theta_nu,trotter_steps, trotter_order, measure='Z'):
    dt = time / trotter_steps

    if trotter_order == 'first':
        dt_substep = dt
    if trotter_order == 'second':
        dt_substep = dt/2
        pauli_terms = pauli_terms + pauli_terms[::-1]
        
    
    qc = QuantumCircuit(N_sites, 1)
    half_N_sites = N_sites // 2
    if theta_nu == 1.74532925E-8 : 
        dt_substep = dt_substep/hbar # since the unitary is divided by hbar in richers test only 
        for i in range(half_N_sites, N_sites):
            qc.x(i) # inital state for the richers test (first half chain up, other half chain down)
    else:
        dt_substep = dt_substep 
        qc.x(range(half_N_sites)) # initial state for rog and vac osc tests


    for _ in range(trotter_steps):
        for coef, pauli in pauli_terms:
            #converts the Pauli operator object into its string representation
            pauli_str = pauli.to_label()
            
            #Creates a list of qubits that are affected by the Pauli term.
            #uses a list comprehension that iterates over the string representation pauli_str of the Pauli operator.
            # enumerate(pauli_str) provides both the index i and the character p for each position in the string.
            #the condition if p != 'I' ensures that only qubits with a Pauli operator other than the identity I are included in the list qubits.
            qubits = [i for i, p in enumerate(pauli_str) if p != 'I']
            
            if len(qubits) == 1:
                #coef * dt_substep: The rotation angle, which is the coefficient coef multiplied by the time step dt_substep
                apply_single_qubit_gate(qc, coef * dt_substep, qubits[0], pauli_str[qubits[0]])
            elif len(qubits) == 2:
                apply_two_qubit_gate(qc, coef * dt_substep, qubits[0], qubits[1], pauli_str[qubits[0]], pauli_str[qubits[1]])
      
      
    # transform basis from Z (up/down or |0>/|1>) to X (equal super position of |up>/ |down> to get the x basis =|+>/|->) using hadamard on the first qubit(b/c we are measurng the first qubit only)
    # the measurement of counts in x basis is done later in main script to compute <sigma_x>
    if measure == 'X':
        qc.h(0)
        
    # prepare the quantum state by transforming basis from Z to Y using sdg on the first qubit(b/c we are measurng the first qubit only)
    # Similarly, the 'Sdg' gate (S-dagger) is applied, which corresponds to the inverse of the phase gate 'S'. This introduces a phase shift of - pi/2.
    # Then, a Hadamard gate 'H' is applied. This sequence of gates transforms from Z basis to Y basis. 
    
    # the measurement of counts in Y basis is done later in main script to compute <sigma_y>
    elif measure == 'Y':
        qc.sdg(0)
        qc.h(0)          
        
    qc.measure(0, 0)
    return qc






