#hamiltonian.py
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Pauli, Operator
from constants import hbar, c , eV, MeV, GeV, G_F, kB



def construct_pert_hamiltonian(omega, B_pert, N_sites):
    pauli_terms = []

    for i in range(N_sites):
        # Construct single-qubit Pauli terms for each site
        Xi = Pauli(f'{"I"*i}X{"I"*(N_sites-i-1)}')
        Yi = Pauli(f'{"I"*i}Y{"I"*(N_sites-i-1)}')
        Zi = Pauli(f'{"I"*i}Z{"I"*(N_sites-i-1)}')

        pauli_terms.append((B_pert[0], Xi))
        pauli_terms.append((B_pert[1], Yi))
        pauli_terms.append((B_pert[2], Zi))

    return pauli_terms


def apply_single_qubit_gate(qc, coef, qubit):
    """
    Applies only an Ry rotation to the specified qubit.
    coef determines the rotation angle.
    """
    qc.rx(2 * coef * 1e-6, qubit)  # all qubits perturbed with Rx
    qc.ry(2 * coef * 1e-6, qubit)  # all qubits perturbed with Ry
    qc.rz(2 * coef * 1e-6, qubit)  # all qubits perturbed with Ry


def pert_circuit(qc, dt_substep, omega, B_pert, N_sites, measure='Z'):
    pauli_terms = construct_pert_hamiltonian(omega, B_pert, N_sites)

    # Apply Ry rotation to every qubit for each perturbation term
    for coef, _ in pauli_terms:
        for q in range(N_sites):
            # apply_single_qubit_gate(qc, coef * dt_substep, q)
            apply_single_qubit_gate(qc, coef, q)



    # for q in range(N_sites):
    #     apply_single_qubit_gate(qc, alpha, B_pert, q)
    # Basis transformation for measurement
    if measure == 'X':
        qc.h(range(N_sites))
    elif measure == 'Y':
        qc.sdg(range(N_sites))
        qc.h(range(N_sites))

    return qc
