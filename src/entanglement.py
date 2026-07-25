# entanglement.py
#
# Pure RDM -> scalar functionals used by the exact-state analysis path. Each takes
# a Qiskit DensityMatrix (or any object exposing a .data array) for some subsystem
# and returns a real number. These are only meaningful on the exact state (see
# exact_state.py); there is no tractable sampled estimator for them.

from itertools import product

import numpy as np

from qiskit.quantum_info import Pauli


def renyi2_from_rdm(rdm, eps=1e-15):
    """
    Second Renyi entropy S_2 = -log2(Tr rho^2) of a reduced density matrix.

    Inputs:
    - rdm: DensityMatrix (or object with .data) for the subsystem of interest.
    - eps: floor on the purity before the log, to avoid log2(0).

    Output:
    - S_2 as a float (bits). 0 for a pure state, log2(dim) for maximally mixed.
    """
    rho = np.asarray(rdm.data, dtype=complex)
    purity = np.real(np.trace(rho @ rho))
    purity = np.clip(purity, eps, 1.0)
    return -np.log2(purity)


def stabilizer_renyi_magic_from_rdm(rdm, eps=1e-15, return_pauli_weights=False):
    """
    Stabilizer 2-Renyi magic M_2 of a density matrix.

    M_2 = -log2( (1/dim) * sum_P |Tr(rho P)|^4 ), summed over all Pauli strings P
    on the subsystem. Quantifies non-stabilizerness ("magic"). Cost is 4^n in the
    number of qubits n of the RDM, so this is only used on small subsystems / the
    small full system of the exact-state tests.

    Inputs:
    - rdm: DensityMatrix (or object with .data). Its dimension must be a power of 2.
    - eps: floor on the magic argument before the log.
    - return_pauli_weights: if True, also return the Pauli-weight distribution
      (index k = summed |Tr(rho P)|^2 over Pauli strings with k non-identity
      factors, normalized by dim).

    Output:
    - magic as a float, or (magic, pauli_weight_distribution) if
      return_pauli_weights is True.
    """
    rho = np.asarray(rdm.data, dtype=complex)
    dim = rho.shape[0]
    n_qubits = int(np.round(np.log2(dim)))

    if 2 ** n_qubits != dim:
        raise ValueError(f"RDM dimension {dim} is not a power of 2.")

    pauli_labels = ["I", "X", "Y", "Z"]

    sum_cP4 = 0.0
    pauli_weight_distribution = np.zeros(n_qubits + 1, dtype=float)

    for label_tuple in product(pauli_labels, repeat=n_qubits):
        label = "".join(label_tuple)

        weight = sum(ch != "I" for ch in label)

        P = Pauli(label).to_matrix()
        cP = np.trace(rho @ P)

        abs_cP2 = np.abs(cP) ** 2
        abs_cP4 = np.abs(cP) ** 4

        sum_cP4 += abs_cP4
        pauli_weight_distribution[weight] += abs_cP2

    magic_value = sum_cP4 / dim
    magic_value = max(np.real_if_close(magic_value).item(), eps)
    magic = -np.log2(magic_value)

    pauli_weight_distribution = pauli_weight_distribution / dim

    if return_pauli_weights:
        return float(magic), pauli_weight_distribution

    return float(magic)
