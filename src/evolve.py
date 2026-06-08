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
    else:
        raise ValueError(
            f"Unsupported Pauli combination: ({pauli1}, {pauli2}). "
            "Only ('X','X'), ('Y','Y'), and ('Z','Z') are allowed."
        )
        
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
    
def cartan_two_qubit_gate(qc, coef, qubit1, qubit2):
    """
    Implements the optimized quantum circuit for exp(-i * θ/2 * (X⊗X + Y⊗Y + Z⊗Z))
    using the minimal number of CNOTs and single-qubit gates DOI: 10.1103/PhysRevD.108.083039
    """
    theta = (2 * coef)
    
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
        # Qiskit's rXX / rYY / rZZ implement exp(-i * θ/2 * σ⊗σ)
        # Our Hamiltonian uses coef * (σ ⊗ σ)
        # → To match exp(-i * coef * σ⊗σ * t), we need θ = 2 * coef * t
        # Additionally, depending on normalization conventions (e.g. spin operators vs Pauli),
        # an extra factor of 2 may appear. Hence overall factor 4 here.
        julia_like_coef = 4 * coef
        print(
            f"H 2-site term: pair={pair}, "
            f"ops=(SzSz,SpSm,SmSp), coef={julia_like_coef}, "
            f"angle_factor={angle_factor}"
        )
def pauli_label_to_qiskit_ops(pauli, n_qubits):
    """
    Convert a Pauli string into a list of (qubit_index, operator) pairs.

    Inputs:
    - pauli: A Pauli operator (e.g., from Qiskit or a string like "IXYZ").
      It represents a tensor product of single-qubit Pauli operators.
    - n_qubits: Total number of qubits in the system.

    Output:
    - ops: A list of tuples (qubit, op), where:
        * qubit is the integer index of the qubit the operator acts on
        * op is one of 'X', 'Y', or 'Z'
      Identity operators ('I') are ignored.

    Note:
    Qiskit uses big-endian ordering for Pauli labels:
    - The leftmost character acts on qubit (n_qubits - 1)
    - The rightmost character acts on qubit 0
    """
    label = pauli.to_label() if hasattr(pauli, "to_label") else str(pauli)
    ops = []
    for label_pos, op in enumerate(label):
        if op != "I":
            qubit = n_qubits - 1 - label_pos
            ops.append((qubit, op))
    return ops


def apply_qubit_permutation(qc, current_particle_ids, target_particle_ids):
    """
    Apply SWAP gates to reorder qubits so that their particle IDs match a desired layout.

    Inputs:
    - qc: A Qiskit QuantumCircuit. SWAP gates will be appended to this circuit.
    - current_particle_ids: Array-like of length N, where index = qubit position
      and value = particle ID currently stored at that qubit.
      (i.e., current mapping: qubit → particle)
    - target_particle_ids: Array-like of length N specifying the desired mapping:
      at each qubit position, which particle ID should be present.

    Output:
    - qc: The same QuantumCircuit with SWAP operations added to implement the permutation.
    - updated_particle_ids: NumPy array giving the final mapping (qubit → particle)
      after applying the swaps. This should match target_particle_ids.

    Behavior:
    - The function iteratively swaps qubits so that each qubit position contains
      the correct particle ID as specified by target_particle_ids.
    - It updates the internal bookkeeping (current_particle_ids) alongside the circuit.
    """
    # Think of this as physically shuffling qubits so particle labels line up with a target ordering.
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

def apply_one_timestep_dynamic_positions(
    qc,
    τ,
    N, x, Δp, L, shape_name, omega, B,
    n_qubits, Δx, p, geometric_name,
    trotter_steps, trotter_order, periodic,
    particle_ids,
    energy_sign,
    advection,
):
    """
    Apply one full physical timestep τ, but update particle positions
    after every Trotter substep dt = τ / trotter_steps.

    Returns updated:
        qc, N, x, p, omega, particle_ids
    """
        
    """
    Inputs:
    - qc: Qiskit QuantumCircuit to which gates are appended.
    - τ: Total physical time for this timestep.

    Hamiltonian parameters:
    - N: Number of particles/modes.
    - x: Positions array.
    - Δp: Momentum spacing.
    - L: System size.
    - shape_name: Initial distribution shape.
    - omega, B: Physical Hamiltonian parameters.

    Discretization / system:
    - n_qubits: Number of qubits (sites).
    - Δx: Spatial lattice spacing (should satisfy Δx = L / n_qubits).
    - p: Momentum values.
    - geometric_name: Geometry type.
    - periodic: Whether boundary conditions are periodic.

    Trotterization:
    - trotter_steps: Number of Trotter steps.
    - trotter_order: 'first' (Lie-Trotter) or 'second' (Suzuki-Trotter).

    Output:
    - qc: QuantumCircuit with evolution gates for one timestep appended.
    """

    # τ = total physical time for this full step
    # trotter_steps splits τ into smaller steps:
    # dt = time per Trotter step
    dt = τ / trotter_steps

    for step in range(trotter_steps):
        print(f"\n=== Dynamic-position Trotter step {step} ===")

        # -------------------------------------------------
        # 1. Build Hamiltonian using current positions x
        # -------------------------------------------------
        pauli_terms = construct_hamiltonian(
            N, x, Δp, L, shape_name, omega, B,
            n_qubits, Δx, p, geometric_name, periodic
        )

        # dt_substep = effective time used in each exponential
        # - first order: full dt per term
        # - second order: half-step per term (due to symmetric decomposition)
        if trotter_order == "first":
            dt_substep = dt / hbar
            angle_factor_for_print = τ / hbar

        elif trotter_order == "second":
            dt_substep = dt / (2 * hbar)
            pauli_terms = pauli_terms + pauli_terms[::-1]
            angle_factor_for_print = τ / (2 * hbar)

        else:
            raise ValueError("trotter_order must be 'first' or 'second'")

        # -------------------------------------------------
        # 2. Apply quantum evolution for this small dt
        # -------------------------------------------------
        for term_id, (coef, pauli) in enumerate(pauli_terms):
            active_ops = pauli_label_to_qiskit_ops(pauli, n_qubits)
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

        if advection:

            # -------------------------------------------------
            # 3. Move particles by the smaller time dt
            # -------------------------------------------------
            p_mod, p_hat = momentum(p, n_qubits)
            p_hat_x = np.asarray([sub_array[0] for sub_array in p_hat])

            x = x + p_hat_x * c * dt

            if periodic:
                x = np.mod(x, L)
                assert np.all(x >= 0) and np.all(x <= L)

            # -------------------------------------------------
            # 4. Resort metadata after particle motion
            # -------------------------------------------------
            perm = np.argsort(x)

            x_new = np.asarray(x)[perm].copy()
            p_new = np.asarray(p)[perm].copy()
            N_new = np.asarray(N)[perm].copy()
            omega_new = np.asarray(omega)[perm].copy()
            particle_ids_new = particle_ids[perm].copy()
            energy_sign_new = np.asarray(energy_sign)[perm].copy()

            # -------------------------------------------------
            # 5. Permute quantum state to match new sorted order
            # -------------------------------------------------
            qc, particle_ids_after = apply_qubit_permutation(
                qc,
                particle_ids,
                particle_ids_new
            )

            x = x_new
            p = p_new
            N = N_new
            omega = omega_new
            particle_ids = particle_ids_after
            energy_sign = energy_sign_new
    # print_julia_like_hamiltonian(pauli_terms, n_qubits, angle_factor_for_print)
    return qc, N, x, p, omega, particle_ids, energy_sign