# time_evolution.py
#
# Shared time-stepping driver for the test scripts. It owns the boilerplate that
# is identical across every test: the initial sort into position order, the
# base-circuit construction, the per-step stdout progress table, and the loop that
# advances the state with apply_one_timestep_dynamic_positions. Every file except
# run_parameters.json is written by the recorders in params["observables"]
# (observables.py).

import os

import numpy as np

from base_circuit import initialize_base_circuit
from evolve import apply_one_timestep_dynamic_positions
from run_parameters import write_run_parameters
from momentum import vacuum_frequency
from activate_backend import backend_supports_direct_state
from meas_counts import get_direct_state
from observables import Observable


def run_time_evolution(params):
    """
    Drive the time-evolution loop shared by all test scripts.

    The driver sorts the sites into position order (the convention used by every
    test), builds the base circuit, and steps through `times`. On each step it
    calls every recorder in params["observables"], prints one progress row, and
    advances the state with apply_one_timestep_dynamic_positions.

    The driver has no return value. Its outputs are run_parameters.json and the
    recorders' files, all written into params["datadir"]. A test that needs these
    histories reads them back from those files.

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

    # Create data files
    # Fail if direct observables requested from non-direct backend
    supports_direct_state = backend_supports_direct_state(params["backend"])
    for obs in observables:
        assert supports_direct_state or not obs.requires_direct_state, (
            f"{type(obs).__name__} reads the exact state, which only an Aer "
            f"simulator provides. Remove it from params['observables'] or set "
            f"backend_name = 'aer'."
        )
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
    bit_list = bit_list[perm0].tolist()
    print("initial sorted bit_list =", bit_list)

    # the state the recorders read and apply_one_timestep_dynamic_positions
    # advances; omega is the vacuum oscillation frequency, derived from Δm², |p|
    # and energy_sign
    state = {
        "x": x[perm0].copy(),
        "p": p[perm0].copy(),
        "N": N[perm0].copy(),
        "energy_sign": energy_sign[perm0].copy(),
        "particle_ids": particle_ids[perm0].copy(),
        "omega": vacuum_frequency(
            params["delta_m_squared"], p[perm0], energy_sign[perm0]
        ),
        "qc_base": initialize_base_circuit(
            n_qubits=N_sites,
            bit_list_sorted=bit_list,
            B_pert=params["B_pert"],
            alpha=params["alpha"],
        ),
        "direct_state": None,
    }
    # progress table: iteration, cumulative trotter steps, t, then the columns the
    # recorders contribute
    header = "".join(obs.progress_header for obs in observables)
    print(f"{'iter':>6}{'trotter':>9}{'t':>15}{header}")

    for step_idx, t in enumerate(times):
        # -------------------------------
        # Per-step measurement / analysis / output
        # -------------------------------
        state["step_idx"] = step_idx
        state["t"] = t

        # the exact state is pulled once per step and shared by the recorders that
        # need it
        if any(obs.requires_direct_state for obs in observables) and backend_supports_direct_state(params["backend"]):
            state["direct_state"] = get_direct_state(params, state)

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

        state = apply_one_timestep_dynamic_positions(params, state)
