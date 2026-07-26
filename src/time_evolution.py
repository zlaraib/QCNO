# time_evolution.py
#
# Shared time-stepping driver for the test scripts. It owns the boilerplate that
# is identical across every test: the initial sort into position order, the
# base-circuit construction, the recording of positions/momenta (x, px, py, pz)
# every step, the per-step stdout progress table, and the loop that advances the
# state with apply_one_timestep_dynamic_positions. The measurement and observable
# output belong to the recorders in params["observables"] (observables.py).

import os

import numpy as np

from base_circuit import initialize_base_circuit
from evolve import apply_one_timestep_dynamic_positions
from run_parameters import write_run_parameters
from momentum import vacuum_frequency
from activate_backend import backend_supports_direct_state
from meas_counts import get_direct_state
from observables import Observable, reorder_sorted_to_original, reset_file, write_row


def run_time_evolution(params):
    """
    Drive the time-evolution loop shared by all test scripts.

    The driver sorts the sites into position order (the convention used by every
    test), builds the base circuit, and steps through `times`. On each step it
    records the positions/momenta (x, px, py, pz, in original particle order),
    calls every recorder in params["observables"], prints one progress row, and
    advances the state with apply_one_timestep_dynamic_positions.

    The driver has no return value. Its outputs are run_parameters.json, the
    position/momentum files, and the recorders' files, all written into
    params["datadir"]. A test that needs these histories reads them back from
    those files.

    Inputs:
    - params: dict of the run's inputs (the output of a test's
      initialize_parameters), including the list of Observable recorders in
      params["observables"] and the output directory params["datadir"].
    """
    times = params["times"]
    N_sites = params["N_sites"]
    observables = params["observables"]
    datadir = params["datadir"]
    os.makedirs(datadir, exist_ok=True)

    for obs in observables:
        obs.create_file(params)

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
    # start the position/momentum files empty
    x_path = os.path.join(datadir, "t_xsiteval.dat")
    px_path = os.path.join(datadir, "t_pxsiteval.dat")
    py_path = os.path.join(datadir, "t_pysiteval.dat")
    pz_path = os.path.join(datadir, "t_pzsiteval.dat")
    reset_file(x_path)
    reset_file(px_path)
    reset_file(py_path)
    reset_file(pz_path)

    # progress table: iteration, cumulative trotter steps, t, then the columns the
    # recorders contribute
    header = "".join(obs.progress_header for obs in observables)
    print(f"{'iter':>6}{'trotter':>9}{'t':>15}{header}")

    for step_idx, t in enumerate(times):
        # -------------------------------
        # Positions and momenta in original particle order (every sim)
        # -------------------------------
        write_row(x_path, t, reorder_sorted_to_original(x, particle_ids))
        write_row(px_path, t, reorder_sorted_to_original(p[:, 0], particle_ids))
        write_row(py_path, t, reorder_sorted_to_original(p[:, 1], particle_ids))
        write_row(pz_path, t, reorder_sorted_to_original(p[:, 2], particle_ids))

        # -------------------------------
        # Per-step measurement / analysis / output
        # -------------------------------
        # the exact state is pulled once per step and shared by the recorders that
        # need it
        exact_state = None
        if any(obs.requires_exact_state for obs in observables) and backend_supports_direct_state(params["backend"]):
            _, exact_state = get_direct_state(
                qc_base, params["backend"], params["optimization_level"]
            )

        state = {
            "step_idx": step_idx,
            "t": t,
            "qc_base": qc_base,
            "x": x,
            "p": p,
            "N": N,
            "omega": omega,
            "energy_sign": energy_sign,
            "particle_ids": particle_ids,
            "exact_state": exact_state,
        }
        progress = []
        for obs in observables:
            progress += obs.append_row(params, state)

        # Recording happens before the step is advanced, so iteration k has had k
        # full timesteps applied.
        trotter_total = step_idx * params["trotter_steps"]
        width = Observable.PROGRESS_COLUMN_WIDTH
        values = "".join(f"{v:>+{width}.4f}" for v in progress)
        print(f"{step_idx:>6}{trotter_total:>9}{t:>15.6e}{values}")

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
