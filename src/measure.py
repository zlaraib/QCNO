# measure.py
#
# Sampled single-site Pauli measurement — the real-hardware path shared by every
# test. For each requested basis it runs the circuit through meas_counts (which
# rotates that basis into the computational basis and measures every qubit), then
# reduces the shot counts to a per-site Pauli expectation value with
# calc_mean_and_sigma. This is the only measurement route that works on real
# backends; exact-state quantities (entropy, multi-site magic) live in
# exact_state.py and require Aer.

import numpy as np

from meas_counts import meas_counts
from sigma_statistics import calc_mean_and_sigma


def measure(qc_base, params):
    """
    Sample single-site Pauli expectation values for each requested basis.

    Inputs:
    - qc_base: QuantumCircuit for the current timestep, before measurement.
    - params: run-parameter dict. Keys used here:
        "measure": list of Pauli bases to sample, e.g. ["Z"] or ["X", "Y", "Z"].
        Plus the keys meas_counts / calc_mean_and_sigma read from params
        ("N_sites", "backend", "backend_name", "optimization_level", "shots",
        "df").

    Output:
    - sigmas: dict {basis: array of per-site <sigma_basis>} in sorted-site order.
      The [::-1] reversal for Qiskit's little-endian bit order is applied here.
      The caller reorders to original particle order with
      reorder_sorted_to_original when it wants original order (some tests keep
      sorted order deliberately).
    """
    sigmas = {}
    for basis in params["measure"]:
        counts, _ = meas_counts(qc_base, basis, params)
        sig, _ = calc_mean_and_sigma(counts, params)
        sigmas[basis] = np.asarray(sig)[::-1]
    return sigmas
