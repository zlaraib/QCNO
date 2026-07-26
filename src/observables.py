# observables.py
#
# Per-step observable recorders for the time-evolution driver, plus the file-output
# helpers they and the driver share. One recorder = one output file = one quantity.
# A test lists the recorders it wants in params["observables"]; the driver calls
# create_file on each before the loop and append_row on each every step. Files go
# into params["datadir"].

import io
import os
import zipfile

import numpy as np

from qiskit.quantum_info import Pauli

from meas_counts import meas_counts
from sigma_statistics import calc_mean_and_sigma
from entanglement import rdm_for_sites, renyi2_from_rdm, stabilizer_renyi_magic_from_rdm


def reorder_sorted_to_original(data_sorted, particle_ids):
    """
    Map an array given in sorted-site order back to original particle order.

    Inputs:
    - data_sorted: array indexed by sorted-site slot.
    - particle_ids: original particle id occupying each sorted slot.

    Output:
    - data_original: array indexed by original particle id.
    """
    data_sorted = np.asarray(data_sorted)
    data_original = np.empty_like(data_sorted)
    for sorted_slot, pid in enumerate(particle_ids):
        data_original[pid] = data_sorted[sorted_slot]
    return data_original


def reset_file(path):
    """Truncate (or create) a file so a run starts from empty."""
    open(path, "w").close()


def assert_density_matrix_method(backend):
    """
    Require an Aer backend running the density_matrix method.

    Entropies and magic are nonlinear in rho, so they are only meaningful on the
    true state. Any other method applies a noise model by quantum trajectories:
    the saved density matrix is then an average over shots (a Monte-Carlo estimate
    at 1024 shots, and a single pure trajectory at 1 shot), which silently reports
    the wrong entropy rather than failing.
    """
    method = getattr(backend.options, "method", None)
    assert method == "density_matrix", (
        f"entanglement observables need the Aer density_matrix method, got {method!r}. "
        'Pass backend_options=dict(method="density_matrix", ...) to build_backend.'
    )


def write_row(path, t, values):
    """
    Append '<t> <values...>' as one high-precision row (open/append/close).

    Inputs:
    - path: file to append to.
    - t: time, written as the first column.
    - values: 1D sequence of numbers. Pass [value] for a single scalar column.
    """
    with open(path, "a") as f:
        np.savetxt(f, [[t, *values]], fmt="%.16e")


def append_npz(path, name, array):
    """
    Append an array to an .npz as a new member (open/append/close), creating the
    file if needed. An .npz is a zip of .npy members, so this adds to it without
    rewriting or holding in memory what is already there. Read back with
    np.load(path)[name].
    """
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}.npy", buffer.getvalue())


class Observable:
    """
    Template for a per-step observable recorder. Subclasses implement:

    - __init__(...): takes only what identifies the quantity (basis, sites, ...),
      and sets self.progress_header via generate_progress_header.
    - create_file(params): set the output path(s) and truncate them. A recorder
      whose quantities share expensive intermediates (an RDM, a Pauli sum) owns
      several files rather than recomputing them in a second recorder.
    - append_row(params, state): write one row to each file, return one float per
      progress_header column ([] if none).

    A subclass that reads state["exact_state"] sets requires_exact_state = True;
    the driver only pulls the exact state if some recorder asks for it.
    """

    PROGRESS_COLUMN_WIDTH = 13

    def __init__(self):
        self.path = None
        self.progress_header = ""
        self.requires_exact_state = False

    def generate_progress_header(self, *labels):
        """Lay out column labels, one per value append_row returns."""
        return "".join(f"{label:>{self.PROGRESS_COLUMN_WIDTH}}" for label in labels)

    def create_file(self, params):
        """Called once before the time loop."""
        raise NotImplementedError("Observable subclasses must implement create_file")

    def append_row(self, params, state):
        """
        Inputs:
        - params: run-parameter dict.
        - state: the per-step dict built by the driver.

        Output:
        - list of floats, one per column in self.progress_header.
        """
        raise NotImplementedError("Observable subclasses must implement append_row")

    def __repr__(self):
        return type(self).__name__


class SigmaObservable(Observable):
    """
    Sampled single-site <sigma_basis> for one Pauli basis -> t_sigma_<b>.dat, one
    row of '<t> <value per particle...>' per timestep in original particle order.
    This is the shot-count route, the only one that works on real backends. Costs
    one circuit execution per timestep.

    Progress column: the site average of <sigma_basis>.
    """

    def __init__(self, basis):
        """basis: "X", "Y" or "Z"."""
        super().__init__()
        self.basis = basis
        self.progress_header = self.generate_progress_header(
            f"<sigma_{basis.lower()}>"
        )

    def create_file(self, params):
        self.path = os.path.join(params["datadir"], f"t_sigma_{self.basis.lower()}.dat")
        reset_file(self.path)

    def append_row(self, params, state):
        counts, _ = meas_counts(state["qc_base"], self.basis, params)
        sig, _ = calc_mean_and_sigma(counts, params)
        sig = np.asarray(sig)[::-1]  # Qiskit little-endian -> sorted-site order
        sig = reorder_sorted_to_original(sig, state["particle_ids"])

        write_row(self.path, state["t"], sig)
        return [float(np.mean(sig))]

class DensityMatrixObservable(Observable):
    """
    The full density matrix of every timestep -> density_matrices_direct.npz, two
    members per step: rho_<step> holding the 2^N x 2^N complex matrix and
    time_<step> holding its time. Read a run back with

        archive = np.load("density_matrices_direct.npz")
        times = np.array([archive[k] for k in sorted(archive) if k.startswith("time_")])
        rho = np.stack([archive[k] for k in sorted(archive) if k.startswith("rho_")])

    Needs the exact state (Aer only).

    Everything else here is derived from the state and discards it; this keeps it,
    so a run can be re-analysed without re-simulating. It is by far the largest
    output, 4^N complex numbers per step, which is why it is binary rather than the
    ascii every other recorder writes: ~3x smaller before compression, and far more
    than that on a state with a conserved quantum number, where most entries are
    exactly zero. Appending member by member keeps both the memory and the per-step
    cost independent of how long the run is, and leaves the file readable if a run
    is interrupted.

    No progress column.
    """

    def __init__(self):
        super().__init__()
        self.requires_exact_state = True

    def create_file(self, params):
        assert_density_matrix_method(params["backend"])
        self.path = os.path.join(params["datadir"], "density_matrices_direct.npz")

        # a truncated .npz is not a readable one, so start empty by removing it
        if os.path.exists(self.path):
            os.remove(self.path)

    def append_row(self, params, state):
        step_idx = state["step_idx"]
        append_npz(self.path, f"rho_{step_idx:06d}", state["exact_state"].data)
        append_npz(self.path, f"time_{step_idx:06d}", state["t"])
        return []


class DirectSigmaObservable(Observable):
    """
    Exact single-site <sigma_basis> for one Pauli basis -> t_sigma_<b>_direct.dat,
    same row layout as SigmaObservable. Noise-free ground truth read off the
    simulator state, so listing it forces the driver to pull that state and only
    an Aer backend can supply it.

    Progress column: the site average of the exact <sigma_basis>.
    """

    def __init__(self, basis):
        """basis: "X", "Y" or "Z"."""
        super().__init__()
        self.basis = basis
        self.requires_exact_state = True
        self.progress_header = self.generate_progress_header(
            f"<sigma_{basis.lower()}D>"
        )

    def create_file(self, params):
        self.path = os.path.join(
            params["datadir"], f"t_sigma_{self.basis.lower()}_direct.dat"
        )
        reset_file(self.path)

    def append_row(self, params, state):
        exact_state = state["exact_state"]
        N_sites = exact_state.num_qubits

        # qubit q sits at Pauli-label position N_sites-1-q, so sig comes out in
        # sorted-slot order, like the sampled path
        sig = np.zeros(N_sites, dtype=float)
        for q in range(N_sites):
            label = ["I"] * N_sites
            label[N_sites - 1 - q] = self.basis
            val = exact_state.expectation_value(Pauli("".join(label)))
            sig[q] = np.real_if_close(val).real
        sig = reorder_sorted_to_original(sig, state["particle_ids"])

        write_row(self.path, state["t"], sig)
        return [float(np.mean(sig))]

class SingleSiteEntanglementObservable(Observable):
    """
    Per-site Renyi-2 entropy -> t_renyi2_single_direct.dat and stabilizer Renyi-2
    magic -> t_magic_single_direct.dat, both '<t> <value per particle...>'. Both
    come from the same N single-site RDMs, which is why they are one recorder.
    Needs the exact state (Aer only).

    Progress columns: the site averages of S_2 and M_2.
    """

    def __init__(self):
        super().__init__()
        self.requires_exact_state = True
        self.progress_header = self.generate_progress_header("<S2_site>", "<M2_site>")

    def create_file(self, params):
        assert_density_matrix_method(params["backend"])
        self.renyi2_path = os.path.join(params["datadir"], "t_renyi2_single_direct.dat")
        self.magic_path = os.path.join(params["datadir"], "t_magic_single_direct.dat")
        reset_file(self.renyi2_path)
        reset_file(self.magic_path)

    def append_row(self, params, state):
        exact_state = state["exact_state"]
        N_sites = exact_state.num_qubits

        renyi2 = np.zeros(N_sites, dtype=float)
        magic = np.zeros(N_sites, dtype=float)
        for q in range(N_sites):
            rdm = rdm_for_sites(exact_state, [q], N_sites)
            renyi2[q] = renyi2_from_rdm(rdm)
            magic[q] = stabilizer_renyi_magic_from_rdm(rdm)
        renyi2 = reorder_sorted_to_original(renyi2, state["particle_ids"])
        magic = reorder_sorted_to_original(magic, state["particle_ids"])

        write_row(self.renyi2_path, state["t"], renyi2)
        write_row(self.magic_path, state["t"], magic)
        return [float(np.mean(renyi2)), float(np.mean(magic))]


class TwoSiteEntanglementObservable(Observable):
    """
    Renyi-2 entropy -> t_renyi2_two_site_direct.dat and magic ->
    t_magic_two_site_direct.dat of every two-site RDM, one column per site pair
    (i < j) in the order listed in two_site_pairs.dat, which create_file writes.
    Both quantities come from the same RDMs, hence one recorder. Columns stay in
    sorted-site order -- a pair of moving particles has no fixed label. Needs the
    exact state (Aer only).

    Cost is N(N-1)/2 partial traces and 4^2 Pauli terms each per step.

    Progress columns: the pair averages of S_2 and M_2.
    """

    def __init__(self):
        super().__init__()
        self.requires_exact_state = True
        self.progress_header = self.generate_progress_header("<S2_2site>", "<M2_2site>")

    def create_file(self, params):
        assert_density_matrix_method(params["backend"])
        N_sites = params["N_sites"]
        self.pairs = [(i, j) for i in range(N_sites) for j in range(i + 1, N_sites)]

        self.renyi2_path = os.path.join(params["datadir"], "t_renyi2_two_site_direct.dat")
        self.magic_path = os.path.join(params["datadir"], "t_magic_two_site_direct.dat")
        reset_file(self.renyi2_path)
        reset_file(self.magic_path)

        # the column key: row k of this file is the site pair in column k above
        np.savetxt(
            os.path.join(params["datadir"], "two_site_pairs.dat"),
            np.array(self.pairs, dtype=int),
            fmt="%d",
        )

    def append_row(self, params, state):
        exact_state = state["exact_state"]
        N_sites = exact_state.num_qubits

        renyi2 = np.zeros(len(self.pairs), dtype=float)
        magic = np.zeros(len(self.pairs), dtype=float)
        for k, (site_i, site_j) in enumerate(self.pairs):
            rdm = rdm_for_sites(exact_state, [site_i, site_j], N_sites)
            renyi2[k] = renyi2_from_rdm(rdm)
            magic[k] = stabilizer_renyi_magic_from_rdm(rdm)

        write_row(self.renyi2_path, state["t"], renyi2)
        write_row(self.magic_path, state["t"], magic)
        return [float(np.mean(renyi2)), float(np.mean(magic))]


class BipartiteEntanglementObservable(Observable):
    """
    Renyi-2 entropy -> t_renyi2_bipartite_direct.dat and magic ->
    t_magic_bipartite_direct.dat of the first half of the sorted sites, one scalar
    per timestep each. Both come from the same half-system RDM. Needs the exact
    state (Aer only).

    Progress columns: those two scalars.
    """

    def __init__(self):
        super().__init__()
        self.requires_exact_state = True
        self.progress_header = self.generate_progress_header("<S2_bipart>", "<M2_bipart>")

    def create_file(self, params):
        assert_density_matrix_method(params["backend"])
        self.renyi2_path = os.path.join(params["datadir"], "t_renyi2_bipartite_direct.dat")
        self.magic_path = os.path.join(params["datadir"], "t_magic_bipartite_direct.dat")
        reset_file(self.renyi2_path)
        reset_file(self.magic_path)

    def append_row(self, params, state):
        exact_state = state["exact_state"]
        N_sites = exact_state.num_qubits

        rdm = rdm_for_sites(exact_state, range(N_sites // 2), N_sites)
        renyi2 = renyi2_from_rdm(rdm)
        magic = stabilizer_renyi_magic_from_rdm(rdm)

        write_row(self.renyi2_path, state["t"], [renyi2])
        write_row(self.magic_path, state["t"], [magic])
        return [float(renyi2), float(magic)]


class GlobalMagicObservable(Observable):
    """
    Magic of the full state -> t_magic_global_direct.dat (one scalar) and the Pauli
    weight distribution it falls out of -> t_pauli_weight_distribution_direct.dat
    (N+1 columns, weight 0..N; the key is written to pauli_weights.dat). One
    stabilizer_renyi_magic_from_rdm call returns both, which is why they are one
    recorder: that call sums over all 4^N Pauli strings and is the most expensive
    thing in the loop. Needs the exact state (Aer only).

    Progress column: the global magic.
    """

    def __init__(self):
        super().__init__()
        self.requires_exact_state = True
        self.progress_header = self.generate_progress_header("<M2_global>")

    def create_file(self, params):
        assert_density_matrix_method(params["backend"])
        self.magic_path = os.path.join(params["datadir"], "t_magic_global_direct.dat")
        self.weight_path = os.path.join(
            params["datadir"], "t_pauli_weight_distribution_direct.dat"
        )
        reset_file(self.magic_path)
        reset_file(self.weight_path)

        # the column key: weight sector of each column of the distribution
        np.savetxt(
            os.path.join(params["datadir"], "pauli_weights.dat"),
            np.arange(params["N_sites"] + 1, dtype=int),
            fmt="%d",
        )

    def append_row(self, params, state):
        magic, pauli_weights = stabilizer_renyi_magic_from_rdm(
            state["exact_state"], return_pauli_weights=True
        )

        write_row(self.magic_path, state["t"], [magic])
        write_row(self.weight_path, state["t"], pauli_weights)
        return [float(magic)]
