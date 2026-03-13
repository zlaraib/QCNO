#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Pauli, Operator
from constants import hbar, c , eV, MeV, GeV, G_F, kB


def pauli_label_to_qiskit_ops(pauli, n_qubits):
    """
    Convert a Qiskit Pauli label into [(qubit, op), ...]
    respecting Qiskit's bit ordering:
    rightmost character = qubit 0
    """
    label = pauli.to_label() if hasattr(pauli, "to_label") else str(pauli)
    ops = []
    for label_pos, op in enumerate(label):
        if op != "I":
            qubit = n_qubits - 1 - label_pos
            ops.append((qubit, op))
    return ops           

def apply_single_qubit_pert_gate(qc, coef, qubit, pauli):
    if pauli == 'X':
        qc.rx(coef, qubit)
    elif pauli == 'Y':
        qc.ry(coef, qubit)
    elif pauli == 'Z':
        qc.rz( coef, qubit)
        

def pert_circuit(qc, B_pert, N_sites, alpha):
    print("---- PERTURB DEBUG ----")
    print(f"B_pert = {B_pert}")
    print(f"alpha = {alpha}")

    for site in range(N_sites):
        print(f"PERT site={site+1}: coef={B_pert[0]}, op=Sx, local_angle={alpha * B_pert[0]}")
        print(f"PERT site={site+1}: coef={B_pert[1]}, op=Sy, local_angle={alpha * B_pert[1]}")
        print(f"PERT site={site+1}: coef={B_pert[2]}, op=Sz, local_angle={alpha * B_pert[2]}")
        print(f"PERT combined: site={site+1}, angle_factor={alpha}")

        qc.rx(alpha * B_pert[0], site)
        qc.ry(alpha * B_pert[1], site)
        qc.rz(alpha * B_pert[2], site)