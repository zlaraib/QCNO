
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit import QuantumCircuit
from evolve import add_measurement_to_circuit
from qiskit.quantum_info import Statevector,DensityMatrix
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import QiskitRuntimeService

# Abstract ("QIS") gate set handed to the IonQ cloud compiler. IonQ is all-to-all,
# so there is no coupling map to route against; we lower to a generic basis and let
# IonQ's own compiler convert to native gates. This also synthesizes the JJ/BJ
# UnitaryGates into cx + single-qubit rotations so nothing 'unitary' is submitted.
_QIS_BASIS = ["rx", "ry", "rz", "cx", "h", "s", "sdg", "x"]

def get_direct_state(qc_base, backend, optimization_level):
    """
    Transpile a circuit for an Aer simulator backend and retrieve the final
    statevector and density matrix directly from the simulator.

    Inputs:
    - qc_base: QuantumCircuit before save instructions are added.
    - backend: Qiskit Aer simulator backend. This must support
      save_statevector() and save_density_matrix().
    - optimization_level: Transpiler optimization level passed to
      Qiskit's preset pass manager.

    Output:
    - isa_circuit: The transpiled circuit adapted to the target backend.
    - density_matrix: Final density matrix of the simulated circuit.
    - statevector: Final statevector of the simulated circuit.
    """
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()

    assert backend_name is not None and "aer" in backend_name.lower(), (
        "get_direct_state requires a Qiskit Aer simulator backend."
    )

    qc = qc_base.copy()
    qc.save_statevector()
    qc.save_density_matrix()

    # Build a preset pass manager for the chosen backend and optimization level,
    # then transpile the input circuit into a backend-compatible ISA circuit.
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level
    )
    isa_circuit = pm.run(qc)

    result = backend.run(isa_circuit).result()
    data = result.data()

    density_matrix = DensityMatrix(data["density_matrix"])
    statevector = Statevector(data["statevector"])

    print("Statevector retrieved successfully.")
    return isa_circuit, density_matrix, statevector

# This function executes circuits differently depending on the backend type:
# - IonQ → uses backend.run (required for IonQ APIs)
# - Aer → uses Sampler (recommended modern Qiskit interface)
# - IBM → uses backend.run (simple and reliable for hardware/fake backends)
#
# The goal is to ensure compatibility across all backends while using
# the most appropriate execution method for each.

def meas_counts(
    qc_base,
    measure,
    N_sites,
    backend,
    backend_name,
    optimization_level,
    shots,
    two_qubit_gate=None,
    euler_basis=None,
    basis_gates=None,
):
    # two_qubit_gate / euler_basis / basis_gates are accepted for call-signature
    # compatibility with the test notebooks but are no longer used: the JJ/BJ terms
    # are emitted as UnitaryGates (see evolve.py) and the transpiler's built-in
    # UnitarySynthesis pass lowers them to each backend's native two-qubit gate. The
    # old manual TwoQubitBasisDecomposer step this triggered is gone. (Removing these
    # params from the notebook call chain is a separate cleanup.)
    qc = add_measurement_to_circuit(qc_base, N_sites, measure=measure)

    print(f"\n{'=' * 70}")
    print(f"MEASUREMENT BASIS = {measure}")
    print(f"BACKEND NAME      = {backend_name}")
    print(f"{'=' * 70}")

    print("\n[1] LOGICAL CIRCUIT BEFORE BACKEND TRANSPILATION")
    print("Gate counts:")
    print(qc.count_ops())
    print("Logical depth:", qc.depth())

    ibm_backends = {"manila", "ibm", "guadalupe"}
    ionq_backends = {"ionq_simulator", "ionq_noisy_sim", "ionq_qpu"}

    backend_name_attr = getattr(backend, "name", None)
    if callable(backend_name_attr):
        backend_name_attr = backend_name_attr()

    is_aer_backend = (
        backend_name_attr is not None and "aer" in backend_name_attr.lower()
    )

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

    # Common lowering step for every backend: produce the ISA circuit once, then
    # branch below only on how to *execute* it.
    isa_circuit = to_isa(qc)

    print("\n[2] FINAL ISA CIRCUIT")
    print("Gate counts:")
    print(isa_circuit.count_ops())
    print("Final depth:", isa_circuit.depth())

    if is_aer_backend:
        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()

        try:
            counts = result[0].data.meas.get_counts()
        except Exception:
            data_keys = list(result[0].data.keys())
            counts = getattr(result[0].data, data_keys[0]).get_counts()

        print("\n[AER SHOT COUNTS]")
        print(counts)
        return counts, isa_circuit

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

        print("\n[IONQ SHOT COUNTS]")
        print(counts)
        return counts, isa_circuit

    elif backend_name in ibm_backends:
        backend_display_name = getattr(backend, "name", "unknown")
        if callable(backend_display_name):
            backend_display_name = backend_display_name()
        print(f"Backend: {backend_display_name}")

        if backend_name in {"manila", "guadalupe"}:
            result = backend.run(isa_circuit, shots=shots).result()
            counts = result.get_counts()

            print("\n[IBM FAKE BACKEND SHOT COUNTS]")
            print(counts)
            return counts, isa_circuit

        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()

        pub_result = result[0]

        try:
            counts = pub_result.data.meas.get_counts()
        except Exception:
            data_keys = list(pub_result.data.keys())
            counts = getattr(pub_result.data, data_keys[0]).get_counts()

        print("\n[IBM HARDWARE SHOT COUNTS]")
        print(counts)
        return counts, isa_circuit

    else:
        raise ValueError(
            f"Unsupported backend_name={backend_name}. "
            "Expected IBM, IonQ, or Aer backend."
        )
