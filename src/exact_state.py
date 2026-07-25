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
# Site convention (shared with construct_hamiltonian and the sampled sigmas):
# a Pauli label is written left-to-right as qubit N-1 ... qubit 0, so label
# position `site` corresponds to Qiskit qubit index N_sites - 1 - site.

import numpy as np

from qiskit.quantum_info import Pauli, partial_trace

from meas_counts import get_direct_state


class ExactState:
    """
    Wraps the exact statevector / density matrix for one timestep and serves
    exact observables under the source/Julia-style site convention.

    Attributes:
    - statevector, density_matrix: the exact Qiskit objects.
    - N_sites: number of sites (qubits).
    """

    def __init__(self, statevector, density_matrix, N_sites):
        self.statevector = statevector
        self.density_matrix = density_matrix
        self.N_sites = N_sites

    def _site_to_qubit(self, site):
        return self.N_sites - 1 - int(site)

    def single_site_sigmas(self, bases=("X", "Y", "Z")):
        """
        Exact per-site <sigma_b> for each basis in `bases`, in sorted-site order.

        Mirrors measure()'s output shape so a test can compare the exact sigmas
        against the sampled ones site-by-site.
        """
        sigmas = {b: np.zeros(self.N_sites, dtype=float) for b in bases}
        for site in range(self.N_sites):
            for b in bases:
                label = ["I"] * self.N_sites
                label[site] = b
                op = Pauli("".join(label))
                val = self.statevector.expectation_value(op)
                sigmas[b][site] = np.real_if_close(val).real
        return sigmas

    def rdm_for_sites(self, keep_sites):
        """
        Reduced density matrix keeping `keep_sites` and tracing out the rest.

        `keep_sites` are site indices (source/Julia-style); they are mapped to
        Qiskit qubit indices before tracing. The returned RDM feeds the scalar
        functionals in entanglement.py (renyi2_from_rdm,
        stabilizer_renyi_magic_from_rdm), which are invariant to the ordering of
        the kept qubits.
        """
        keep_qubits = {self._site_to_qubit(s) for s in keep_sites}
        traced_out = [q for q in range(self.N_sites) if q not in keep_qubits]
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
    state["direct_ok"] in record_step.
    """
    _, density_matrix, statevector = get_direct_state(
        qc_base, params["backend"], params["optimization_level"]
    )
    return ExactState(statevector, density_matrix, params["N_sites"])
