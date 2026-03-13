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
from perturb import pert_circuit, pauli_label_to_qiskit_ops

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


# def evolve_and_measure_circuit(t, τ, backend_name,backend,optimization_level, N,x,Δp,L, shape_name, omega, B,B_pert,  N_sites, Δx,  p,theta_nu, trotter_steps, trotter_order,periodic, measure='Z'):
def build_evolution_circuit(
    t, τ, backend_name, backend, optimization_level,
    N, x, Δp, L, shape_name, omega, B, B_pert,
    N_sites, Δx, p, theta_nu, trotter_steps, trotter_order, periodic):
    pauli_terms = construct_hamiltonian(N, x,Δp,L, shape_name,omega, B, N_sites, Δx,  p,theta_nu,periodic)
    # print("Pauli Terms:", pauli_terms)
    
    # k = int(round(t / τ))   # number of macro-steps
    # dt = τ / trotter_steps    # micro-step size is fixed forever

    dt = τ 
    
    if trotter_order == 'first':
        dt_substep = dt
    if trotter_order == 'second':
        dt_substep = dt/2
        pauli_terms = pauli_terms + pauli_terms[::-1]
        
    # print_julia_like_hamiltonian(pauli_terms, N_sites, dt_substep, theta_nu)
    qc = QuantumCircuit(N_sites, N_sites)
    half_N_sites = N_sites // 2
    if theta_nu == 1.74532925E-8 : 
        dt_substep = dt_substep/hbar # since the unitary is divided by hbar in richers test only 
        # for i in range(half_N_sites, N_sites):
        #     qc.x(i) # inital state for the richers test (first half chain up, other half chain down)
        bitstr = '1'*half_N_sites + '0'*(N_sites - half_N_sites) # actual initial condition of sherwood
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

    

    qc.barrier(label="after_init")
    alpha= 1e-4
    if B_pert is not None:

        pert_circuit(qc, B_pert, N_sites, alpha)
        qc.barrier(label="after_pert")

    if t > 0:
        # for step in range(trotter_steps):
        #     print(f"\n=== Trotter step {step} ===")
            for term_id, (coef, pauli) in enumerate(pauli_terms):
                active_ops = pauli_label_to_qiskit_ops(pauli, N_sites)
                label = pauli.to_label()
                angle = coef * dt_substep

                # print(
                #     f"H term {term_id}: label={label}, "
                #     f"active_ops={active_ops}, coef={coef}, angle={angle}"
                # )

                if len(active_ops) == 1:
                    qubit, op = active_ops[0]
                    apply_single_qubit_gate(qc, angle, qubit, op)

                elif len(active_ops) == 2:
                    (q0, op0), (q1, op1) = active_ops
                    apply_two_qubit_gate(qc, angle, q0, q1, op0, op1)
    print_julia_like_hamiltonian(pauli_terms, N_sites, dt_substep, theta_nu)
        
    if backend_name == 'aer':
        # Add the save_density_matrix instruction
        qc.save_statevector()
        qc.save_density_matrix()
            # evolution loops...
    qc.barrier(label="after_evolve")
    
    return qc



def add_measurement_to_circuit(qc_base, N_sites, measure='Z'):
    qc = qc_base.copy()

    if measure == 'X':
        qc.h(range(N_sites))
    elif measure == 'Y':
        qc.sdg(range(N_sites))
        qc.h(range(N_sites))
    elif measure != 'Z':
        raise ValueError(f"Unsupported measurement basis: {measure}")

    qc.measure_all()
    qc.barrier(label="after_measure")
    return qc

def print_julia_like_hamiltonian(pauli_terms, N_sites, dt_substep, theta_nu):
    print("\n=== Julia-like grouped Hamiltonian ===")

    pair_groups = {}

    for coef, pauli in pauli_terms:
        active_ops = pauli_label_to_qiskit_ops(pauli, N_sites)

        if len(active_ops) == 2:
            (q0, op0), (q1, op1) = active_ops

            # Qiskit qubit -> physical site (1-indexed)
            s0 = N_sites - q0
            s1 = N_sites - q1
            pair = tuple(sorted((s0, s1)))

            if pair not in pair_groups:
                pair_groups[pair] = {"coef": coef, "ops": []}

            pair_groups[pair]["ops"].append(op0 + op1)

    angle_factor = dt_substep

    for pair in sorted(pair_groups):
        coef = pair_groups[pair]["coef"]
        julia_like_coef = 4 * coef
        print(
            f"H 2-site term: pair={pair}, "
            f"ops=(SzSz,SpSm,SmSp), coef={julia_like_coef}, "
            f"angle_factor={angle_factor}"
        )