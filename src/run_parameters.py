# run_parameters.py
#
# Write the input parameters of a simulation run to a file, for reproducibility.
# Intended to be called once at the entry of a run function, before the time loop,
# so the recorded values are the true inputs (several arrays -- x, p, N, omega,
# particle_ids -- get re-sorted in place during the run).

import os
import json

import numpy as np


def _json_default(v):
    """
    Fallback serializer for json.dump. Converts the non-JSON-native types that show
    up in a run's parameters -- numpy arrays/scalars -- to plain Python, and renders
    anything else (backend objects, functions, ...) as its repr string so the dump
    never fails.
    """
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return repr(v)


def write_run_parameters(path, params):
    """
    Dump a run's input parameters to a JSON file.

    Inputs:
    - path: Destination file path. Parent directories are created if needed.
    - params: Mapping of parameter name -> value. Values may be numpy arrays/scalars
      (converted to plain Python) or non-serializable objects such as the backend
      (recorded as their repr string via _json_default).

    Output:
    - None. Writes `path`, overwriting any existing file.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as f:
        json.dump(dict(params), f, indent=2, sort_keys=True, default=_json_default)
