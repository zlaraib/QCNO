
import os
import csv

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from evolve import add_measurement_to_circuit
from activate_backend import backend_supports_direct_state
from qiskit.quantum_info import Statevector,DensityMatrix

# Abstract ("QIS") gate set handed to the IonQ cloud compiler. IonQ is all-to-all,
# so there is no coupling map to route against; we lower to a generic basis and let
# IonQ's own compiler convert to native gates. This also synthesizes the JJ/BJ
# UnitaryGates into cx + single-qubit rotations so nothing 'unitary' is submitted.
_QIS_BASIS = ["rx", "ry", "rz", "cx", "h", "s", "sdg", "x"]

# Per-call circuit-size log. One row per meas_counts call, written to CSV instead of
# to screen. count_ops() is an OrderedDict; it is stored as a single stringified cell
# (csv quoting handles its internal commas).
_CIRCUIT_LOG_COLUMNS = [
    "iteration",
    "measurement_basis",
    "initial_depth",
    "final_depth",
    "initial_counts",
    "final_counts",
    "shot_counts",
]
# Log paths whose header has already been written in THIS process. A re-run is a
# fresh Python process, so this set starts empty: the first write of a run opens the
# file in "w" mode (overwrite the stale file + write header), and later writes in the
# same run append.
_initialized_circuit_logs = set()

def _log_circuit_info(log_path, row, backend_name=None):
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    first_write = log_path not in _initialized_circuit_logs
    with open(log_path, "w" if first_write else "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if first_write:
            # Backend is constant for the whole run, so record it once at the top of
            # the file rather than as a repeated column on every row.
            f.write(f"# backend: {backend_name}\n")
            writer.writerow(_CIRCUIT_LOG_COLUMNS)
            _initialized_circuit_logs.add(log_path)
        writer.writerow(row)

def get_direct_state(params, state):
    """
    Transpile a circuit for an Aer simulator backend and retrieve the final state
    directly from the simulator.

    The saved representation follows the simulation method: the density_matrix
    method holds a density matrix (the only exact representation of a noisy, mixed
    state, and the only one it accepts a save instruction for), every other method
    holds a statevector, which is 2^N numbers instead of 4^N. Note that a noise
    model applied under a statevector-like method is realized by quantum
    trajectories, so the returned statevector is one trajectory, not the ensemble.

    Inputs:
    - params: run-parameter dict, for the Aer backend and the transpiler
      optimization level.
    - state: the driver's per-step dict, for the circuit to run.

    Output:
    - direct_state: Final DensityMatrix (density_matrix method) or Statevector
      (any other method). Both support expectation_value() and partial_trace().
    """
    assert backend_supports_direct_state(params["backend"]), (
        "get_direct_state requires a Qiskit Aer simulator backend."
    )

    qc = state["qc_base"].copy()
    if getattr(params["backend"].options, "method", None) == "density_matrix":
        qc.save_density_matrix()
    else:
        qc.save_statevector()

    # Build a preset pass manager for the chosen backend and optimization level,
    # then transpile the input circuit into a backend-compatible ISA circuit.
    pm = generate_preset_pass_manager(
        backend=params["backend"],
        optimization_level=params["optimization_level"]
    )
    isa_circuit = pm.run(qc)

    result = params["backend"].run(isa_circuit).result()
    data = result.data()

    if "density_matrix" in data:
        direct_state = DensityMatrix(data["density_matrix"])
    else:
        direct_state = Statevector(data["statevector"])

    return direct_state

# This function executes circuits differently depending on the backend type:
# - IonQ → uses backend.run (required for IonQ APIs)
# - Aer → uses Sampler (recommended modern Qiskit interface)
# - IBM → uses backend.run (simple and reliable for hardware/fake backends)
#
# The goal is to ensure compatibility across all backends while using
# the most appropriate execution method for each.

def meas_counts(qc_base, measure, params, iteration=None):
    # params is the run-parameter dict (see initialize_parameters in the tests). Only
    # the execution-relevant keys are read here. The former two_qubit_gate /
    # euler_basis / basis_gates arguments are gone: the JJ/BJ terms are emitted as
    # UnitaryGates (see evolve.py) and the transpiler's built-in UnitarySynthesis pass
    # lowers them to each backend's native two-qubit gate, so those values were dead.
    #
    # iteration only labels the row this call writes to the circuit-size log; pass the
    # timestep index once it is threaded down from the callers (blank until then).
    N_sites = params["N_sites"]
    backend = params["backend"]
    backend_name = params["backend_name"]
    optimization_level = params["optimization_level"]
    shots = params["shots"]

    qc = add_measurement_to_circuit(qc_base, N_sites, measure=measure)

    ibm_backends = {"manila", "ibm", "guadalupe"}
    ionq_backends = {"ionq_simulator", "ionq_noisy_sim", "ionq_qpu"}

    is_aer_backend = backend_supports_direct_state(backend)

    def to_isa(input_qc):
        """
        Lower input_qc to an executable ("ISA") circuit for the selected backend.

        IonQ: transpile to the abstract _QIS_BASIS and let IonQ's cloud compiler do
        native conversion (all-to-all, so no routing here). Aer / IBM: a
        backend-targeted preset pass manager, which both synthesizes the JJ/BJ
        UnitaryGates and routes/maps to the device topology.
        """
        if backend_name in ionq_backends:
            return transpile(
                input_qc,
                basis_gates=_QIS_BASIS,
                optimization_level=optimization_level,
            )

        if is_aer_backend or backend_name in ibm_backends:
            try:
                pm = generate_preset_pass_manager(
                    backend=backend,
                    optimization_level=optimization_level,
                )
                return pm.run(input_qc)

            except UnboundLocalError as e:
                # Some IBM targets fail target-conversion on a missing frequency;
                # fall back to explicit basis_gates + coupling_map from the config.
                if "frequency" not in str(e):
                    raise

                print("\nWarning: IBM backend.target conversion failed due to missing frequency.")
                print("Using fallback transpilation with basis_gates and coupling_map.")

                config = backend.configuration()
                return transpile(
                    input_qc,
                    basis_gates=getattr(config, "basis_gates", None),
                    coupling_map=getattr(config, "coupling_map", None),
                    optimization_level=optimization_level,
                )

        raise ValueError(
            f"Unsupported backend_name={backend_name}. "
            "Expected IBM, IonQ, or Aer backend."
        )

    # Common lowering step for every backend: produce the ISA circuit once, log its
    # size to file, then branch below only on how to *execute* it.
    isa_circuit = to_isa(qc)

    if is_aer_backend:
        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()

        try:
            counts = result[0].data.meas.get_counts()
        except Exception:
            data_keys = list(result[0].data.keys())
            counts = getattr(result[0].data, data_keys[0]).get_counts()

    elif backend_name in ionq_backends:
        # IonQ rejects high-level / opaque gates; the abstract-basis transpile above
        # must have lowered everything. Fail loudly if any survived.
        unsupported_ionq = {
            "state_preparation",
            "initialize",
            "unitary",
            "u",
            "u3",
        }
        remaining = set(isa_circuit.count_ops()) & unsupported_ionq

        if remaining:
            raise ValueError(f"IonQ circuit still has unsupported gates: {remaining}")

        job = backend.run(isa_circuit, shots=shots)
        counts = job.get_counts()

    elif backend_name in ibm_backends:
        if backend_name in {"manila", "guadalupe"}:
            result = backend.run(isa_circuit, shots=shots).result()
            counts = result.get_counts()

        else:
            sampler = Sampler(mode=backend)
            job = sampler.run([isa_circuit], shots=shots)
            result = job.result()

            pub_result = result[0]

            try:
                counts = pub_result.data.meas.get_counts()
            except Exception:
                data_keys = list(pub_result.data.keys())
                counts = getattr(pub_result.data, data_keys[0]).get_counts()

    else:
        raise ValueError(
            f"Unsupported backend_name={backend_name}. "
            "Expected IBM, IonQ, or Aer backend."
        )

    _log_circuit_info(
        "datafiles/circuit_info.csv",
        [
            iteration,
            measure,
            qc.depth(),
            isa_circuit.depth(),
            qc.count_ops(),
            isa_circuit.count_ops(),
            counts,
        ],
        backend_name=backend_name,
    )
    return counts, isa_circuit

