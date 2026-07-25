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


def run_time_evolution(params, datadir, record_step):
    """
    Drive the time-evolution loop shared by all test scripts.

    The driver sorts the sites into position order (the convention used by every
    test), builds the base circuit, and steps through `times`. On each step it
    records the positions/momenta (x, px, py, pz, in original particle order),
    invokes record_step for the test's own measurement/analysis/output, and then
    advances the state with apply_one_timestep_dynamic_positions.

    The driver has no return value. Its outputs are run_parameters.json and the
    position/momentum files it writes into datadir, plus the per-step call to
    record_step. A test that needs position/momentum histories reads them back
    from those files.

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
    - datadir: directory for the position/momentum files (created if needed).
    - record_step: callback invoked once per timestep as
          record_step(state, params, datadir)
      where state is a dict:
          {"step_idx", "t", "qc_base", "x", "p", "N", "omega", "energy_sign",
           "particle_ids", "direct_ok"}
      with arrays in current sorted-site order. It performs the test's own
      measurement/analysis/output.
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

    for step_idx, t in enumerate(times):
        # -------------------------------
        # Positions and momenta in original particle order (every sim)
        # -------------------------------
        append_row(x_path, t, reorder_sorted_to_original(x, particle_ids))
        append_row(px_path, t, reorder_sorted_to_original(p[:, 0], particle_ids))
        append_row(py_path, t, reorder_sorted_to_original(p[:, 1], particle_ids))
        append_row(pz_path, t, reorder_sorted_to_original(p[:, 2], particle_ids))

        # -------------------------------
        # Test-specific measurement / analysis / output
        # -------------------------------
        record_step(
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
