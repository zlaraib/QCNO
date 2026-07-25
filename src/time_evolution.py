# time_evolution.py
#
# Shared time-stepping driver for the test scripts. It owns the boilerplate that
# is identical across every test: the initial sort into position order, the
# base-circuit construction, the recording of positions/momenta (x, px, py, pz)
# every step, and the loop that advances the state with
# apply_one_timestep_dynamic_positions. Each test supplies a record_step
# callback that performs its own (test-specific) measurement and file output.
#
# File output convention: files are opened in append mode, written, and closed
# on every write (see append_row / append_scalar_row). The per-step compute far
# outweighs the open/close cost, and this keeps callers free of handle
# bookkeeping. Call reset_file once before a run to start from an empty file.

import os

import numpy as np

from base_circuit import initialize_base_circuit
from activate_backend import backend_supports_direct_state
from evolve import apply_one_timestep_dynamic_positions
from run_parameters import write_run_parameters
from momentum import vacuum_frequency
from measure import measure


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


def append_row(path, t, values):
    """Append '<t> <values...>' as one high-precision row (open/append/close)."""
    row = np.concatenate(([t], np.asarray(values, dtype=float).ravel()))
    with open(path, "a") as f:
        np.savetxt(f, row[None, :], fmt="%.16e")


def append_scalar_row(path, t, value):
    """Append '<t> <value>' as one high-precision row (open/append/close)."""
    with open(path, "a") as f:
        np.savetxt(f, np.array([[t, float(value)]]), fmt="%.16e")


def _sigma_path(datadir, basis):
    return os.path.join(datadir, f"t_sigma_{basis.lower()}.dat")


def reset_observable_files(params, datadir):
    """Truncate every per-step observable file record_observables appends to, so a
    run starts from empty (the driver does the same for the position/momentum
    files)."""
    os.makedirs(datadir, exist_ok=True)
    for basis in params["measure"]:
        reset_file(_sigma_path(datadir, basis))


def print_progress_header(params):
    """Print the column header for the per-step progress table that
    record_observables writes to stdout: iteration, trotter_steps, t, then one
    site-averaged <sigma_b> column per basis in params["measure"]."""
    labels = "".join(f"{'<sigma_' + b.lower() + '>':>13}" for b in params["measure"])
    print(f"{'iter':>6}{'trotter':>9}{'t':>15}{labels}")


def record_observables(state, params, datadir):
    """
    Standardized per-step measurement and output. Replaces the per-test
    record_step callback that used to be injected into run_time_evolution: the
    driver now owns this and it is driven entirely by params, with fixed file
    names.

    Inputs:
    - state: the per-step dict the driver builds (keys used: "t", "qc_base",
      "particle_ids", "direct_ok").
    - params: run-parameter dict. Keys used:
        "measure": Pauli bases sampled each step -> t_sigma_<b>.dat, written in
          original particle order.
    - datadir: output directory.

    Quantities derived purely from the sampled sigmas (e.g. the density-matrix
    components rho_ee/mumu/emu, see rho_from_counts.py) are computed in
    post-processing from the t_sigma_*.dat files, not here. Exact-state
    observables (direct sigmas, entanglement entropy, magic — via
    exact_state.reduce_exact_state + entanglement.py), gated by
    state["direct_ok"], are added here when the direct-state tests are ported.
    """
    t = state["t"]
    qc_base = state["qc_base"]
    particle_ids = state["particle_ids"]

    # Sampled single-site Pauli sigmas, written in original particle order.
    sigmas = {
        basis: reorder_sorted_to_original(sig, particle_ids)
        for basis, sig in measure(qc_base, params).items()
    }
    for basis, sig in sigmas.items():
        append_row(_sigma_path(datadir, basis), t, sig)

    # Basic per-step progress row (see print_progress_header for the columns):
    # iteration, cumulative trotter steps applied so far, t, then the domain
    # (site) average of each measured sigma. Recording happens before the step is
    # advanced, so iteration k has had k full timesteps applied.
    trotter_total = state["step_idx"] * params["trotter_steps"]
    means = "".join(f"{np.mean(sigmas[basis]):>+13.4f}" for basis in params["measure"])
    print(f"{state['step_idx']:>6}{trotter_total:>9}{t:>15.6e}{means}")


def run_time_evolution(params, datadir):
    """
    Drive the time-evolution loop shared by all test scripts.

    The driver sorts the sites into position order (the convention used by every
    test), builds the base circuit, and steps through `times`. On each step it
    records the positions/momenta (x, px, py, pz, in original particle order),
    calls record_observables for the standardized per-step measurement/output,
    and then advances the state with apply_one_timestep_dynamic_positions.

    The driver has no return value. Its outputs are run_parameters.json, the
    position/momentum files, and the observable files written by
    record_observables (all into datadir). A test that needs these histories
    reads them back from those files.

    Inputs:
    - params: dict of the run's physics inputs (the output of a test's
      initialize_parameters). Keys used here:
        Physics state (copied, then mutated across steps):
          "times", "x", "p", "N", "energy_sign", "bit_list", "N_sites"
        Vacuum frequency (omega is derived here as Δm²/(2|p|)*energy_sign):
          "delta_m_squared"
        Initial circuit:
          "backend", "B_pert", "alpha"
        Propagation (forwarded to apply_one_timestep_dynamic_positions):
          "tau", "dp", "L", "shape_name", "B", "dx", "geometric_name",
          "trotter_steps", "trotter_order", "periodic", "advection"
        Observables (consumed by record_observables): "measure".
    - datadir: directory for all output files (created if needed).
    """
    times = params["times"]
    N_sites = params["N_sites"]
    backend = params["backend"]
    os.makedirs(datadir, exist_ok=True)

    # record the exact inputs of this run before anything is sorted/mutated
    write_run_parameters(os.path.join(datadir, "run_parameters.json"), params)

    x = np.asarray(params["x"]).copy()
    p = np.asarray(params["p"]).copy()
    N = np.asarray(params["N"]).copy()
    energy_sign = np.asarray(params["energy_sign"]).copy()
    bit_list = np.asarray(params["bit_list"])

    # original physical labels
    particle_ids = np.arange(N_sites)

    # initial sort into position order (shared convention across all tests)
    perm0 = np.argsort(x)
    x = x[perm0].copy()
    p = p[perm0].copy()
    N = N[perm0].copy()
    energy_sign = energy_sign[perm0].copy()
    particle_ids = particle_ids[perm0].copy()
    bit_list = bit_list[perm0].tolist()
    print("initial sorted bit_list =", bit_list)

    # vacuum oscillation frequency, derived from Δm², |p|, and energy_sign
    omega = vacuum_frequency(params["delta_m_squared"], p, energy_sign)

    qc_base = initialize_base_circuit(
        n_qubits=N_sites,
        bit_list_sorted=bit_list,
        B_pert=params["B_pert"],
        alpha=params["alpha"],
    )
    direct_ok = backend_supports_direct_state(backend)

    # start the position/momentum files empty
    x_path = os.path.join(datadir, "t_xsiteval.dat")
    px_path = os.path.join(datadir, "t_pxsiteval.dat")
    py_path = os.path.join(datadir, "t_pysiteval.dat")
    pz_path = os.path.join(datadir, "t_pzsiteval.dat")
    reset_file(x_path)
    reset_file(px_path)
    reset_file(py_path)
    reset_file(pz_path)

    # start the per-step observable files empty too
    reset_observable_files(params, datadir)

    print_progress_header(params)

    for step_idx, t in enumerate(times):
        # -------------------------------
        # Positions and momenta in original particle order (every sim)
        # -------------------------------
        append_row(x_path, t, reorder_sorted_to_original(x, particle_ids))
        append_row(px_path, t, reorder_sorted_to_original(p[:, 0], particle_ids))
        append_row(py_path, t, reorder_sorted_to_original(p[:, 1], particle_ids))
        append_row(pz_path, t, reorder_sorted_to_original(p[:, 2], particle_ids))

        # -------------------------------
        # Standardized per-step measurement / analysis / output
        # -------------------------------
        record_observables(
            {
                "step_idx": step_idx,
                "t": t,
                "qc_base": qc_base,
                "x": x,
                "p": p,
                "N": N,
                "omega": omega,
                "energy_sign": energy_sign,
                "particle_ids": particle_ids,
                "direct_ok": direct_ok,
            },
            params,
            datadir,
        )

        # stop after recording the last point
        if step_idx == len(times) - 1:
            break

        qc_base, N, x, p, omega, particle_ids, energy_sign = apply_one_timestep_dynamic_positions(
            qc_base,
            params["tau"],
            N, x, params["dp"], params["L"], params["shape_name"], omega, params["B"],
            N_sites, params["dx"], p, params["geometric_name"],
            params["trotter_steps"], params["trotter_order"], params["periodic"],
            particle_ids,
            energy_sign,
            params["advection"],
        )
