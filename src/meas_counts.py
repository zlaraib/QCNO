
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import ECRGate, IGate, RZGate, SXGate, XGate, CXGate
from qiskit import transpile 
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from evolve import add_measurement_to_circuit
from qiskit.quantum_info import Statevector,DensityMatrix
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import RXXGate

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

    def decompose_to_requested_basis(input_qc, label):
        if two_qubit_gate is None or euler_basis is None or basis_gates is None:
            print(f"\n[2] BASIC QIS TRANSPILATION FOR {label}")
            out_qc = transpile(
                input_qc,
                basis_gates=["rx", "ry", "rz", "cx", "h", "s", "sdg", "x"],
                optimization_level=optimization_level,
            )
            print("Gate counts:")
            print(out_qc.count_ops())
            print("Depth after basic QIS transpilation:", out_qc.depth())
            return out_qc

        if "ms" in basis_gates:
            raise ValueError(
                "basis_gates contains 'ms'. This function does not support "
                "IonQ native MS submission. Use ionq_simulator/noisy_sim with RXX "
                "or build a separate native-gate circuit."
            )

        gate_map = {
            "ECR": ECRGate(),
            "CX": CXGate(),
            "RXX": RXXGate(np.pi / 2),
        }

        if two_qubit_gate not in gate_map:
            raise ValueError(
                f"Unsupported two_qubit_gate={two_qubit_gate}. "
                f"Allowed values are {list(gate_map.keys())}."
            )
        # decomposes all the 2 qubit gates in the circuit for 1 time step and optimizes all those 2-qubit gates as a whole for each time step. 

        # Decompose two-qubit gates using KAK decomposition
        # Gate	Use Case (Two-qubit gate to be used in the KAK decomposition):
        # CXGate()	Standard choice for IBM Quantum devices (most universal).
        # ECRGate()	More optimized for IBM devices that natively support Echoed Cross Resonance.
        # iSwapGate()	Used in other superconducting architectures (e.g., Google Sycamore).
        # RXXGate()	Needed when simulating parametric two-qubit interactions.
        # Available Euler Bases in Qiskit: Valid options are ['ZYZ', 'ZXZ', 'XYX', 'U', 'U3', 'U1X', 'PSX', 'ZSX', 'RR'].
        # 'U3'	Uses the U3 gate decomposition (IBM's standard, equivalent to any single-qubit unitary).
        # 'ZYZ'	Uses RZ-RY-RZ decomposition, commonly used in quantum computing.
        # 'ZXZ'	Uses RZ-RX-RZ decomposition, less common but useful in some applications.
        # 'XYX'	Uses RX-RY-RX decomposition, an alternative single-qubit Euler representation

        # Decompose two-qubit gates using the Euler basis and two-qubit gate
        # selected in initialize_parameters() for the chosen backend.

        decomposer = TwoQubitBasisDecomposer(
            gate_map[two_qubit_gate],
            euler_basis=euler_basis,
        )

        # Create a new quantum circuit to hold the decomposed gates
        new_qc = QuantumCircuit(*input_qc.qregs, *input_qc.cregs)

        for inst in input_qc.data:
            gate = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            new_qargs = [new_qc.qubits[input_qc.find_bit(q).index] for q in qargs]
            new_cargs = [new_qc.clbits[input_qc.find_bit(c).index] for c in cargs]

            if gate.num_qubits == 2:
                unitary_matrix = Operator(gate).data
                decomposed_circuit = decomposer(unitary_matrix)
                new_qc.append(decomposed_circuit.to_instruction(), new_qargs, [])
            else:
                new_qc.append(gate, new_qargs, new_cargs)

        out_qc = transpile(
            new_qc,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

        print(f"\n[2] AFTER CUSTOM BASIS TRANSPILATION FOR {label}")
        print(f"two_qubit_gate = {two_qubit_gate}")
        print(f"euler_basis    = {euler_basis}")
        print(f"basis_gates    = {basis_gates}")
        print("Gate counts:")
        print(out_qc.count_ops())
        print("Depth after custom basis transpilation:", out_qc.depth())

        return out_qc

    if is_aer_backend:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        isa_circuit = pm.run(qc)

        print("\n[2] FINAL AER ISA CIRCUIT")
        print("Gate counts:")
        print(isa_circuit.count_ops())
        print("Final Aer depth:", isa_circuit.depth())

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
        if backend_name == "ionq_qpu":
            qc = transpile(
                qc,
                basis_gates=["rx", "ry", "rz", "cx", "h", "s", "sdg", "x"],
                optimization_level=optimization_level,
            )

            print("\n[2] IONQ QPU QIS-SAFE TRANSPILATION")
            print("Gate counts:")
            print(qc.count_ops())
            print("Depth after IonQ QPU transpilation:", qc.depth())

        else:
            qc = decompose_to_requested_basis(qc, label="IONQ SIMULATOR")

        isa_circuit = qc

        print("\n[3] FINAL IONQ SUBMITTED CIRCUIT")
        print("Gate counts:")
        print(isa_circuit.count_ops())
        print("Final IonQ submitted depth:", isa_circuit.depth())

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
        qc = decompose_to_requested_basis(qc, label="IBM")

        try:
            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=optimization_level,
            )
            isa_circuit = pm.run(qc)

        except UnboundLocalError as e:
            if "frequency" not in str(e):
                raise

            print("\nWarning: IBM backend.target conversion failed due to missing frequency.")
            print("Using fallback transpilation with basis_gates and coupling_map.")

            config = backend.configuration()
            backend_basis_gates = getattr(config, "basis_gates", None)
            coupling_map = getattr(config, "coupling_map", None)

            isa_circuit = transpile(
                qc,
                basis_gates=backend_basis_gates,
                coupling_map=coupling_map,
                optimization_level=optimization_level,
            )

        backend_display_name = getattr(backend, "name", "unknown")
        if callable(backend_display_name):
            backend_display_name = backend_display_name()

        print("\n[3] FINAL IBM ISA CIRCUIT")
        print(f"Backend: {backend_display_name}")
        print("Gate counts:")
        print(isa_circuit.count_ops())
        print("Final IBM hardware depth:", isa_circuit.depth())

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