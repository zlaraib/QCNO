# base_circuit.py
#
# Build the initial state-preparation circuit shared by the test drivers.

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from perturb import pert_circuit


def initialize_base_circuit(n_qubits, bit_list_sorted=None, B_pert=None, alpha=None):
    """
    Build the base circuit that prepares the initial computational-basis state,
    optionally followed by a perturbation.

    Inputs:
    - n_qubits: Number of qubits / sites.
    - bit_list_sorted: List of '0'/'1' characters, one per site in sorted order.
      The bitstring is reversed before state preparation to match Qiskit's
      little-endian qubit ordering (site i -> qubit n_qubits-1-i).
    - B_pert, alpha: Optional perturbation field and strength. When B_pert is
      not None, pert_circuit is applied after the initial state is prepared.

    Output:
    - qc: QuantumCircuit with the initial state prepared (and barriered), plus
      the perturbation if one was requested.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    if bit_list_sorted is None:
        raise ValueError("bit_list_sorted must be provided")

    bitstr = "".join(bit_list_sorted)
    print("init bit_list_sorted =", bit_list_sorted)
    print("init bitstr =", bitstr)

    # Reverse for Qiskit's little-endian convention
    prep = StatePreparation(bitstr[::-1])
    qc.append(prep, qargs=range(n_qubits))
    qc.barrier(label="after_init")

    if B_pert is not None:
        if pert_circuit is None:
            raise ImportError("B_pert was provided, but pert_circuit could not be imported.")
        pert_circuit(qc, B_pert, n_qubits, alpha)
        qc.barrier(label="after_pert")

    return qc
