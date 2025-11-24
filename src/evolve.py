# evolve.py 

import numpy as np
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit 
from qiskit.circuit.library import ECRGate, IGate, RZGate, SXGate, XGate, CXGate
from qiskit import transpile 
from numpy import pi

from qiskit.quantum_info import Pauli, Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import StatePreparation

from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB
from geometric_func import geometric_func
from hamiltonian import construct_hamiltonian
from perturb import pert_circuit

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
        
def custom_rzz(qc, theta, qubit1, qubit2):
    # qc.barrier()
    qc.cx(qubit1, qubit2)
    # qc.barrier()
    qc.rz(theta, qubit2)
    # qc.barrier()
    qc.cx(qubit1, qubit2)

def custom_rxx(qc, theta, qubit1, qubit2):
    # qc.barrier()
    qc.h(qubit1)
    # qc.barrier()
    qc.h(qubit2)
    custom_rzz(qc, theta, qubit1, qubit2)
    # qc.barrier()
    qc.h(qubit1)
    # qc.barrier()
    qc.h(qubit2)

def custom_ryy(qc, theta, qubit1, qubit2):
    # qc.barrier()
    qc.sdg(qubit1)
    # qc.barrier()
    qc.sdg(qubit2)
    # qc.barrier()
    qc.h(qubit1)
    # qc.barrier()
    qc.h(qubit2)
    custom_rzz(qc, theta, qubit1, qubit2)
    # qc.barrier()
    qc.h(qubit1)
    # qc.barrier()
    qc.h(qubit2)
    # qc.barrier()
    qc.s(qubit1)
    # qc.barrier()
    qc.s(qubit2)

def apply_custom_two_qubit_gate(qc, coef, qubit1, qubit2, pauli1, pauli2):
    angle = 2 * coef  # Calculate the rotation angle
    if pauli1 == 'X' and pauli2 == 'X':
        custom_rxx(qc, angle, qubit1, qubit2)
    elif pauli1 == 'Y' and pauli2 == 'Y':
        custom_ryy(qc, angle, qubit1, qubit2)
    elif pauli1 == 'Z' and pauli2 == 'Z':
        custom_rzz(qc, angle, qubit1, qubit2)

def cartan_two_qubit_gate(qc, coef, qubit1, qubit2, pauli1, pauli2):
    """
    Implements the optimized quantum circuit for exp(-i * θ/2 * (X⊗X + Y⊗Y + Z⊗Z))
    using the minimal number of CNOTs and single-qubit gates as shown in Fig. 3 DOI: 10.1103/PhysRevD.108.083039
    """
    theta = (2 * coef)
    # if (pauli1 == 'Z' and pauli2 == 'Z') or (pauli1 == 'Y' and pauli2 == 'Y') or (pauli1 == 'X' and pauli2 == 'X'): #how to place these? is this source of error?
        # First CNOT
    qc.cx(qubit1, qubit2)
    # First rotation gates
    qc.rx(theta, qubit1)
    qc.rz(theta, qubit2)
    
    # Middle Hadamard
    qc.h(qubit1)

    # Second CNOT
    qc.cx(qubit1, qubit2)
    # Middle S gate
    qc.s(qubit1)

    # Reverse Z rotation from earlier
    qc.rz(-theta, qubit2)
    
    # Reverse Hadamard gate
    qc.h(qubit1)

    # Final CNOT
    qc.cx(qubit1, qubit2)
    
    # Final X rotations
    qc.rx(pi / 2, qubit1)
    qc.rx(-pi / 2, qubit2)
    
def cartan_two_qubit_gate(qc, coef, qubit1, qubit2):
    """
    Implements the optimized quantum circuit for exp(-i * θ/2 * (X⊗X + Y⊗Y + Z⊗Z))
    using the minimal number of CNOTs and single-qubit gates DOI: 10.1103/PhysRevD.108.083039
    """
    theta = (2 * coef) + pi/2
    
    # Barrier before starting the operation to group the whole sequence
    qc.barrier()

    # First CNOT
    qc.cx(qubit1, qubit2)

    # Barrier after the first CNOT to isolate the next set of operations
    qc.barrier()

    # First rotation gates
    qc.rx(theta, qubit1)
    qc.rz(theta, qubit2)
    
    # Barrier after the rotations
    qc.barrier()

    # Middle Hadamard
    qc.h(qubit1)

    # Barrier before the second CNOT to separate logical operations
    qc.barrier()

    # Second CNOT
    qc.cx(qubit1, qubit2)
    
    # Middle S gate
    qc.s(qubit1)

    # Barrier before reversing the operations
    qc.barrier()

    # Reverse Z rotation from earlier
    qc.rz(-theta, qubit2)
    
    # Reverse Hadamard gate
    qc.h(qubit1)

    # Barrier before final CNOT to keep the final operations isolated
    qc.barrier()

    # Final CNOT
    qc.cx(qubit1, qubit2)
    
    # Final X rotations
    qc.rx(np.pi / 2, qubit1)
    qc.rx(-np.pi / 2, qubit2)

    # Final barrier to mark the end of this block of operations
    qc.barrier()



def optimized_two_qubit_circuit(qc, theta, qubit1, qubit2): #has no cnot min error in entaglement
    """Integrates the optimized two-qubit circuit into the given quantum circuit."""
    # Apply the Uq gate with theta = -pi/2 and phi = pi/2 to first qubit. 10.1103/PhysRevD.107.023007
    qc.u(-pi/2, 0, 0, qubit1)
    
    # Apply the RZZ gate to generate the ZZ gate in the paper with theta = pi/2
    qc.rzz(pi/2, qubit1, qubit2)
    
    # Apply the Uq gate with theta = pi/2 and phi = pi to first qubit
    qc.u(pi/2, pi/2, -pi/2, qubit1)
    
    # Apply the Uq gate on the second qubit with theta = 2*theta
    qc.u(2*theta, 0, 0, qubit2)
    
    # Apply the Rz gate with appropriate lambdas
    qc.rz(2*theta - 3*pi/2, qubit1)
    qc.rz(-3*pi/2, qubit2)
    
    # Apply the RZZ gate with theta = pi/2
    qc.rzz(pi/2, qubit1, qubit2)
    
    # Apply the Uq gate again to first qubit
    qc.u(-pi/2, 0, 0, qubit1)
    qc.u(2*theta, 0, 0, qubit2)
    
    # Apply the Rz gate with lambda = -pi/2 to second qubit
    qc.rz(-pi/2, qubit2)
    
    # Apply the RZZ gate with theta = pi/2
    qc.rzz(pi/2, qubit1, qubit2)
    
    # Apply the final Uq gate to first qubit
    qc.u(pi/2, pi/2, -pi/2, qubit1)


def evolve_and_measure_circuit(time, backend_name,backend,optimization_level, N,x,Δp,L, shape_name, omega, B,B_pert,  N_sites, Δx,  p,theta_nu, trotter_steps, trotter_order,periodic, measure='Z'):
    pauli_terms = construct_hamiltonian(N, x,Δp,L, shape_name,omega, B, N_sites, Δx,  p,theta_nu,periodic)
    # print("Pauli Terms:", pauli_terms)
    
    dt = time / trotter_steps

    if trotter_order == 'first':
        dt_substep = dt
    if trotter_order == 'second':
        dt_substep = dt/2
        pauli_terms = pauli_terms + pauli_terms[::-1]
        
    
    qc = QuantumCircuit(N_sites, N_sites)
    half_N_sites = N_sites // 2
    if theta_nu == 1.74532925E-8 : 
        dt_substep = dt_substep/hbar # since the unitary is divided by hbar in richers test only 
        # for i in range(half_N_sites, N_sites):
        #     qc.x(i) # inital state for the richers test (first half chain up, other half chain down)
        bitstr = '0'*half_N_sites + '1'*(N_sites - half_N_sites) # actual initial condition of sherwood
        prep = StatePreparation(bitstr[::-1])  # reverse due to qubit order in Qiskit
        qc.append(prep, qargs=range(N_sites))

    else:
        dt_substep = dt_substep 
        if theta_nu == 1000 : # for Josh test
            # inital state for the Josh test (first half chain + 1 up, other half -1 chain down)
            # String labels: Labels like '01' can initialize the qubits, where '01' means qubit 0 is |1⟩ (down) and qubit 1 is |0⟩ (up).
            bitstr = '0'*(half_N_sites+ 1) + '1'*(N_sites - half_N_sites - 1)
            print("bitstr = ",bitstr )
            prep = StatePreparation(bitstr[::-1])  
            # prep = StatePreparation(bitstr)  
            print("prep=", prep)
            qc.append(prep, qargs=range(N_sites)) # Append the StatePreparation gate to the circuit on all N_sites qubits
        else:
            qc.x(range(half_N_sites)) # initial state for rog and vac osc tests

    
    if B_pert is not None:

        # Apply the perturbation only at the first step
        pert_circuit(qc, dt_substep, omega, B_pert, N_sites, measure=measure)
        # print("After pert qc", qc)

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
                    # print(f"Applying term: {coef} * {pauli.to_label()} on qubits {qubits}")
                    # apply_custom_two_qubit_gate(qc, coef * dt_substep, qubits[0], qubits[1], pauli_str[qubits[0]], pauli_str[qubits[1]])
                    # cartan_two_qubit_gate(qc, (coef * dt_substep)/N_sites, qubits[0], qubits[1])
                    # optimized_two_qubit_circuit(qc,- coef * dt_substep/N_sites, qubits[0], qubits[1])
                    # Insert SWAP gate after the two-qubit gate
                    # qc.swap(qubits[0], qubits[1])

      
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
    
    if backend_name == 'aer':
        # Add the save_density_matrix instruction
        qc.save_density_matrix()
    qc.measure_all()  # Measure all qubits.
    return qc



