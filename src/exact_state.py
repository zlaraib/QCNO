# exact_state.py
#
# Exact-state extraction path. Where measure.py samples single-site Pauli
# expectation values from shot counts (the real-hardware route), this pulls the
# full statevector / density matrix directly from the simulator and exposes exact
# single-site sigmas plus reduced density matrices for arbitrary site subsets.
# This is the ONLY route to genuinely many-body quantities (entanglement entropy,
# multi-site / global magic), and it requires an Aer simulator: get_direct_state
# asserts an Aer backend, so callers should invoke this only when
# backend_supports_direct_state(backend) is True (the driver exposes that as
# state["direct_ok"]).
#
# Site convention: site index == sorted slot == Qiskit qubit index, the same
# indexing the sampled sigma path (measure / calc_mean_and_sigma) produces, so
# exact and sampled single-site quantities line up slot-for-slot before the
# caller reorders to original particle order. A single-site Pauli acting on qubit
# q is written at Pauli-label position N_sites - 1 - q (Qiskit's little-endian
# label order).

import numpy as np

from qiskit.quantum_info import Pauli, partial_trace

from meas_counts import get_direct_state


class ExactState:
    """
    Wraps the exact statevector / density matrix for one timestep and serves
    exact observables in sorted-slot (= Qiskit qubit) order.

    Attributes:
    - statevector, density_matrix: the exact Qiskit objects.
    - N_sites: number of sites (qubits).
    """

    def __init__(self, statevector, density_matrix, N_sites):
        self.statevector = statevector
        self.density_matrix = density_matrix
        self.N_sites = N_sites

    def single_site_sigmas(self, bases=("X", "Y", "Z")):
        """
        Exact per-site <sigma_b> for each basis in `bases`, indexed by sorted slot
        (= Qiskit qubit). Mirrors measure()'s output shape so a test can compare
        the exact sigmas against the sampled ones slot-for-slot.
        """
        sigmas = {b: np.zeros(self.N_sites, dtype=float) for b in bases}
        for q in range(self.N_sites):
            for b in bases:
                label = ["I"] * self.N_sites
                label[self.N_sites - 1 - q] = b  # qubit q -> label position N-1-q
                val = self.statevector.expectation_value(Pauli("".join(label)))
                sigmas[b][q] = np.real_if_close(val).real
        return sigmas

    def rdm_for_sites(self, keep_sites):
        """
        Reduced density matrix keeping `keep_sites` (sorted-slot / qubit indices)
        and tracing out the rest. The returned RDM feeds the scalar functionals
        in entanglement.py (renyi2_from_rdm, stabilizer_renyi_magic_from_rdm),
        which are invariant to the ordering of the kept qubits.
        """
        keep = {int(s) for s in keep_sites}
        traced_out = [q for q in range(self.N_sites) if q not in keep]
        return partial_trace(self.density_matrix, traced_out)


def reduce_exact_state(qc_base, params):
    """
    Build an ExactState for the current timestep from the exact simulator state.

    Inputs:
    - qc_base: QuantumCircuit for the current timestep, before measurement.
    - params: run-parameter dict. Keys used: "backend", "optimization_level",
      "N_sites".

    Output:
    - ExactState wrapping the exact statevector and density matrix.

    Requires an Aer backend (get_direct_state asserts this). Guard the call with
    state["direct_ok"] in record_observables.
    """
    _, density_matrix, statevector = get_direct_state(
        qc_base, params["backend"], params["optimization_level"]
    )
    return ExactState(statevector, density_matrix, params["N_sites"])
