# evolve.py 

# from math import perm

import numpy as np
from scipy.linalg import expm

from qiskit.quantum_info import Pauli
from qiskit.circuit.library import UnitaryGate

from momentum import momentum
from constants import hbar, c
from hamiltonian import construct_hamiltonian

# XX + YY + ZZ, precomputed once (real, symmetric 4x4 matrix).
_H_JJ = Pauli('XX').to_matrix() + Pauli('YY').to_matrix() + Pauli('ZZ').to_matrix()

# Single-qubit Pauli matrices, for building per-site Hamiltonians.
_X = Pauli('X').to_matrix()
_Y = Pauli('Y').to_matrix()
_Z = Pauli('Z').to_matrix()

def apply_JJ_gate(qc, coef, qubit1, qubit2):
    """
    Emit the isotropic two-site coupling exp(-i * coef * (X⊗X + Y⊗Y + Z⊗Z)) as a
    single 2-qubit UnitaryGate.

    The gate is left un-decomposed here on purpose: the transpiler synthesizes it
    into the backend's native two-qubit gate via its built-in UnitarySynthesis pass
    (see meas_counts). coef is the per-substep angle (coef * dt_substep from
    apply_one_timestep_dynamic_positions); rxx(2*coef) = exp(-i*coef*X⊗X), and X⊗X,
    Y⊗Y, Z⊗Z mutually commute so their product is this single exponential.
    X⊗X + Y⊗Y + Z⊗Z is swap-symmetric, so qubit order is irrelevant.
    """
    U = expm(-1j * coef * _H_JJ)
    qc.append(UnitaryGate(U, label='JJ'), [qubit1, qubit2])

def apply_single_site_gate(qc, B, qubit):
    """
    Emit a full single-site term exp(-i * (B[0]*X + B[1]*Y + B[2]*Z)) as one
    1-qubit UnitaryGate, where B is the per-axis angle vector already scaled by
    dt_substep in apply_one_timestep_dynamic_positions.

    This groups a site's X, Y, Z rotations into a single exact single-qubit
    unitary. X, Y, Z do not mutually commute, so emitting them as separate
    rx/ry/rz rotations would itself be an intra-site Trotter split; the
    exponential of the summed generator here avoids that.
    """
    U = expm(-1j * (B[0] * _X + B[1] * _Y + B[2] * _Z))
    qc.append(UnitaryGate(U, label='BJ'), [qubit])


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

def print_julia_like_hamiltonian(terms, n_qubits, angle_factor):
    print("\n=== Julia-like grouped Hamiltonian ===")

    for coef, op, qubits in terms:
        if op == 'JJ':
            q0, q1 = qubits
            # Julia uses 1-based site labels; qubit q maps back to site n_qubits - q.
            s0 = n_qubits - q0
            s1 = n_qubits - q1
            pair = tuple(sorted((s0, s1)))

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

def apply_one_timestep_dynamic_positions(params, state):
    """
    Apply one full physical timestep τ, but update particle positions
    after every Trotter substep dt = τ / trotter_steps.

    Inputs:
    - params: run-parameter dict, for the timestep τ ("tau"), the Hamiltonian
      ("dp", "L", "shape_name", "B", "N_sites", "dx", "geometric_name",
      "periodic"), the Trotterization ("trotter_steps", "trotter_order") and
      whether particles move ("advection").
    - state: the driver's per-step dict, read for "qc_base", "N", "x", "p",
      "omega", "particle_ids" and "energy_sign".

    Output:
    - state, with those same seven entries advanced by one timestep.
    """
    # τ = total physical time for this full step
    # trotter_steps splits τ into smaller steps:
    # dt = time per Trotter step
    dt = params["tau"] / params["trotter_steps"]

    for step in range(params["trotter_steps"]):
        # -------------------------------------------------
        # 1. Build Hamiltonian using current positions x
        # -------------------------------------------------
        pauli_terms = construct_hamiltonian(params, state)

        # dt_substep = effective time used in each exponential
        # - first order: full dt per term
        # - second order: half-step per term (due to symmetric decomposition)
        if params["trotter_order"] == "first":
            dt_substep = dt / hbar
            angle_factor_for_print = params["tau"] / hbar

        elif params["trotter_order"] == "second":
            dt_substep = dt / (2 * hbar)
            pauli_terms = pauli_terms + pauli_terms[::-1]
            angle_factor_for_print = params["tau"] / (2 * hbar)

        else:
            raise ValueError("trotter_order must be 'first' or 'second'")

        # -------------------------------------------------
        # 2. Apply quantum evolution for this small dt
        # -------------------------------------------------
        for term_id, (coef, op, qubits) in enumerate(pauli_terms):
            angle = coef * dt_substep

            # print(
            #     f"H term {term_id}: op={op}, qubits={qubits}, "
            #     f"coef={coef}, angle={angle}"
            # )

            if op == 'JJ':
                apply_JJ_gate(state["qc_base"], angle, qubits[0], qubits[1])
            elif op == 'BJ':
                apply_single_site_gate(state["qc_base"], angle, qubits[0])
            else:
                raise ValueError(f"Unknown Hamiltonian term op: {op!r}")

        if params["advection"]:

            # -------------------------------------------------
            # 3. Move particles by the smaller time dt
            # -------------------------------------------------
            p_mod, p_hat = momentum(state["p"], params["N_sites"])
            p_hat_x = np.asarray([sub_array[0] for sub_array in p_hat])

            state["x"] = state["x"] + p_hat_x * c * dt

            if params["periodic"]:
                state["x"] = np.mod(state["x"], params["L"])
                assert np.all(state["x"] >= 0) and np.all(state["x"] <= params["L"])

            # -------------------------------------------------
            # 4. Resort metadata after particle motion
            # -------------------------------------------------
            perm = np.argsort(state["x"])
            particle_ids_new = state["particle_ids"][perm].copy()

            # -------------------------------------------------
            # 5. Permute quantum state to match new sorted order
            # -------------------------------------------------
            state["qc_base"], state["particle_ids"] = apply_qubit_permutation(
                state["qc_base"],
                state["particle_ids"],
                particle_ids_new
            )

            state["x"] = np.asarray(state["x"])[perm].copy()
            state["p"] = np.asarray(state["p"])[perm].copy()
            state["N"] = np.asarray(state["N"])[perm].copy()
            state["omega"] = np.asarray(state["omega"])[perm].copy()
            state["energy_sign"] = np.asarray(state["energy_sign"])[perm].copy()
    # print_julia_like_hamiltonian(pauli_terms, params["N_sites"], angle_factor_for_print)
    return state
