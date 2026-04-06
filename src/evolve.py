# evolve.py 

# from math import perm

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


def add_measurement_to_circuit(qc_base, n_qubits, measure='Z'):
    qc = qc_base.copy()

    if measure == 'X':
        qc.h(range(n_qubits))
    elif measure == 'Y':
        qc.sdg(range(n_qubits))
        qc.h(range(n_qubits))
    elif measure != 'Z':
        raise ValueError(f"Unsupported measurement basis: {measure}")

    qc.measure_all()
    qc.barrier(label="after_measure")
    return qc

def print_julia_like_hamiltonian(pauli_terms, n_qubits, angle_factor):
    print("\n=== Julia-like grouped Hamiltonian ===")

    pair_groups = {}

    for coef, pauli in pauli_terms:
        active_ops = pauli_label_to_qiskit_ops(pauli, n_qubits)

        if len(active_ops) == 2:
            (q0, op0), (q1, op1) = active_ops
            s0 = n_qubits - q0
            s1 = n_qubits - q1
            pair = tuple(sorted((s0, s1)))

            if pair not in pair_groups:
                pair_groups[pair] = {"coef": coef, "ops": []}

            pair_groups[pair]["ops"].append(op0 + op1)

    for pair in sorted(pair_groups):
        coef = pair_groups[pair]["coef"]
        julia_like_coef = 4 * coef
        print(
            f"H 2-site term: pair={pair}, "
            f"ops=(SzSz,SpSm,SmSp), coef={julia_like_coef}, "
            f"angle_factor={angle_factor}"
        )
def pauli_label_to_qiskit_ops(pauli, n_qubits):
    """
    Qiskit Pauli labels are big-endian:
    leftmost char -> qubit n_qubits-1
    rightmost char -> qubit 0
    """
    label = pauli.to_label() if hasattr(pauli, "to_label") else str(pauli)
    ops = []
    for label_pos, op in enumerate(label):
        if op != "I":
            qubit = n_qubits - 1 - label_pos
            ops.append((qubit, op))
    return ops

def initialize_base_circuit(
    n_qubits, theta_nu, energy_sign_sorted, B_pert=None, alpha=None
):
    qc = QuantumCircuit(n_qubits, n_qubits)
    half_n = n_qubits // 2

    if theta_nu == 1.74532925E-8:
        # Up = |0>, Down = |1>
        bit_list = ['0' if s == -1 else '1' for s in energy_sign_sorted]
        bitstr = ''.join(bit_list)

        print("init energy_sign_sorted =", energy_sign_sorted)
        print("init bitstr =", bitstr)

        prep = StatePreparation(bitstr[::-1])  # Qiskit endianness
        qc.append(prep, qargs=range(n_qubits))

    else:
        if theta_nu == 1000:
            bitstr = '1' * (half_n + 1) + '0' * (n_qubits - half_n - 1)
            prep = StatePreparation(bitstr[::-1])
            qc.append(prep, qargs=range(n_qubits))
        else:
            qc.x(range(half_n))

    qc.barrier(label="after_init")

    if B_pert is not None:
        pert_circuit(qc, B_pert, n_qubits, alpha)
        qc.barrier(label="after_pert")

    return qc

def apply_qubit_permutation(qc, current_particle_ids, target_particle_ids):
    """
    Reorder the live quantum state so qubit positions match target_particle_ids.
    Both inputs are arrays of particle IDs indexed by qubit position.
    """
    current_particle_ids = list(current_particle_ids)
    target_particle_ids = list(target_particle_ids)

    for target_pos in range(len(target_particle_ids)):
        wanted_particle = target_particle_ids[target_pos]
        current_pos = current_particle_ids.index(wanted_particle)

        if current_pos != target_pos:
            qc.swap(target_pos, current_pos)
            current_particle_ids[target_pos], current_particle_ids[current_pos] = (
                current_particle_ids[current_pos],
                current_particle_ids[target_pos],
            )

    return qc, np.array(current_particle_ids)

def apply_one_timestep(
    qc,
    τ,
    N, x, Δp, L, shape_name, omega, B,
    n_qubits, Δx, p, theta_nu, trotter_steps, trotter_order, periodic
):
    # Metadata are assumed already sorted and aligned with qubit order
    pauli_terms = construct_hamiltonian(
        N, x, Δp, L, shape_name, omega, B,
        n_qubits, Δx, p, theta_nu, periodic
    )

    dt = τ / trotter_steps

    if trotter_order == 'first':
        dt_substep = dt / hbar
        angle_factor_for_print = τ / hbar
    elif trotter_order == 'second':
        dt_substep = dt / (2 * hbar)
        angle_factor_for_print = τ / (2 * hbar)
        pauli_terms = pauli_terms + pauli_terms[::-1]
    else:
        raise ValueError("trotter_order must be 'first' or 'second'")

    for step in range(trotter_steps):
        print(f"\n=== Trotter step {step} ===")

        for term_id, (coef, pauli) in enumerate(pauli_terms):
            active_ops = pauli_label_to_qiskit_ops(pauli, n_qubits)
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

    # print_julia_like_hamiltonian(pauli_terms, n_qubits, angle_factor_for_print)
    return qc