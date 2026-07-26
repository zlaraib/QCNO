# entanglement.py
#
# Reduced density matrices of the exact state, and the scalar functionals built on
# them. Each functional takes a Qiskit DensityMatrix (or any object exposing a
# .data array) for some subsystem and returns a real number. These are only
# meaningful on the exact state (state["statevector"] / state["density_matrix"] in
# the driver); there is no tractable sampled estimator for them.

from itertools import product

import numpy as np

from qiskit.quantum_info import Pauli, partial_trace


def rdm_for_sites(state, keep_sites, N_sites):
    """
    Reduced density matrix of a subsystem.

    Inputs:
    - state: exact Statevector or DensityMatrix of the full system.
    - keep_sites: sorted-slot (= Qiskit qubit) indices to keep.
    - N_sites: number of sites (qubits) in the full system.

    Output:
    - DensityMatrix of the kept sites, with the rest traced out.
    """
    keep = {int(s) for s in keep_sites}
    traced_out = [q for q in range(N_sites) if q not in keep]
    return partial_trace(state, traced_out)


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

    TODO: this is the pure-state definition, and it is applied here to RDMs and to
    noisy (mixed) states. sum_P |Tr(rho P)|^2 = dim * Tr(rho^2), so for mixed rho
    the 1/dim normalization is short by the purity and the result picks up
    mixedness on top of magic -- a maximally mixed qubit gives M_2 = 1 instead of
    0. Normalizing by sum_P |Tr(rho P)|^2 instead of dim would fix that limit
    (equivalently, subtract the Renyi-2 entropy), but even then stabilizer Renyi
    entropies are not magic monotones for mixed states. Look into which measure the
    noise study actually wants before reading these numbers as magic. The same
    normalization question applies to pauli_weight_distribution below.

    Inputs:
    - rdm: DensityMatrix or Statevector (Tr(rho P) and <psi|P|psi> are the same
      sum). Its dimension must be a power of 2.
    - eps: floor on the magic argument before the log.
    - return_pauli_weights: if True, also return the Pauli-weight distribution
      (index k = summed |Tr(rho P)|^2 over Pauli strings with k non-identity
      factors, normalized by dim).

    Output:
    - magic as a float, or (magic, pauli_weight_distribution) if
      return_pauli_weights is True.
    """
    n_qubits = rdm.num_qubits
    dim = 2 ** n_qubits

    pauli_labels = ["I", "X", "Y", "Z"]

    sum_cP4 = 0.0
    pauli_weight_distribution = np.zeros(n_qubits + 1, dtype=float)

    for label_tuple in product(pauli_labels, repeat=n_qubits):
        label = "".join(label_tuple)

        weight = sum(ch != "I" for ch in label)

        cP = rdm.expectation_value(Pauli(label))

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
