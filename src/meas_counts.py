
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
    two_qubit_gate,
    euler_basis,
    basis_gates,
):
    qc = add_measurement_to_circuit(qc_base, N_sites, measure=measure)

    # print("\nOriginal Circuit:")
    # print(qc.draw())
    # Print the gate counts in the original circuit
    gate_counts = qc.count_ops()
    # print("\nGate counts in the original circuit:")
    print(gate_counts)

    # # Circuit depth = The maximum number of gates (or operations) applied sequentially on any qubit in the circuit. This is equivalent to the longest "path" of operations in the circuit.
    # # Depth provides a measure of how many time steps are required to execute the circuit i.e. layers of gates reuired in the circuit to execute from start to finish.
    print("Circuit depth= ", qc.depth())

    # qc = transpile(qc, basis_gates=NoiseModel.from_backend(backend).basis_gates, coupling_map=backend.configuration().coupling_map, optimization_level=optimization_level)

    ibm_backends = {"manila", "ibm", "guadalupe"}
    ionq_backends = {"ionq", "ionq_noisy_sim", "ionq_native", "ionq_qpu"}

    backend_name_attr = getattr(backend, "name", None)
    if callable(backend_name_attr):
        backend_name_attr = backend_name_attr()

    is_aer_backend = (
        backend_name_attr is not None and "aer" in backend_name_attr.lower()
    )

    if backend_name in ibm_backends or backend_name in ionq_backends:
        if two_qubit_gate is None:
            raise ValueError(
                f"backend_name={backend_name} requires two_qubit_gate "
                "to be passed from initialize_parameters()."
            )

        if euler_basis is None or basis_gates is None:
            raise ValueError(
                f"backend_name={backend_name} requires euler_basis and basis_gates "
                "to be passed from initialize_parameters()."
            )

        gate_map = {
            "ECR": ECRGate(),
            "CX": CXGate(),
        }

        if two_qubit_gate not in gate_map:
            raise ValueError(
                f"Unsupported two_qubit_gate={two_qubit_gate}. "
                f"Allowed values are {list(gate_map.keys())}."
            )

        basis_gate_obj = gate_map[two_qubit_gate]

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
            basis_gate_obj,
            euler_basis=euler_basis,
        )

        # Create a new quantum circuit to hold the decomposed gates
        new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

        for inst in qc.data:
            gate = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            # map old bits to positions in the new circuit
            new_qargs = [new_qc.qubits[qc.find_bit(q).index] for q in qargs]
            new_cargs = [new_qc.clbits[qc.find_bit(c).index] for c in cargs]

            if gate.num_qubits == 2:
                unitary_matrix = Operator(gate).data
                decomposed_circuit = decomposer(unitary_matrix)
                new_qc.append(decomposed_circuit.to_instruction(), new_qargs, [])
            else:
                new_qc.append(gate, new_qargs, new_cargs)

        # # Print the decomposed circuit
        # print("\nDecomposed Circuit:")
        # # print(new_qc.draw())
        # # Print the gate counts in the decomposed circuit
        # gate_counts = new_qc.count_ops()
        # print("\nGate counts in the decomposed circuit:")
        # print(gate_counts)
        # print("Circuit depth in decomposed circuit= ", new_qc.depth())
        # print(backend.configuration().basis_gates)

        # Transpile the new circuit using the basis gates selected in
        # initialize_parameters() for the chosen backend.
        # For IBM, this may be basis_gates=['rz', 'sx', 'x', 'cx', 'u3'].
        # For IonQ, this may be basis_gates=['rx', 'ry', 'rxx', 'ms'].
        transpiled_circuit = transpile(
            new_qc,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

        qc = transpiled_circuit

        # # qc = transpile(qc, backend=backend, basis_gates=['rx', 'ry', 'rz', 'cx']) #added basis gateset for testing the noise model, can be removed for any case really 
        # qc = transpile(qc, backend=backend)
        # print(qc.draw())

        # Print the gate counts in the transpiled circuit
        # gate counts in the transpiled circuit 
        gate_counts = qc.count_ops()
        print("\nGate counts in the transpiled circuit:")
        print(gate_counts)
        print("Circuit depth in transpiled circuit= ", qc.depth())

    if is_aer_backend:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        isa_circuit = pm.run(qc)

        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]

        counts = result[0].data.meas.get_counts()
        # result = backend.run(isa_circuit).result()
        return counts, isa_circuit

    elif backend_name in ionq_backends:
        job = backend.run(qc, shots=shots)

        # def execute_circuit(circuit):
        #     # Submit job to your backend
        #     job = backend.run(circuit, shots=shots)
        #     result = job.result()
        #     counts = result.get_counts()

        #     # Optional: compute observable (example: parity or Z expectation)
        #     # Here's a simple observable: expectation value of Z on qubit 0
        #     z_exp = 0
        #     for bitstring, count in counts.items():
        #         sign = 1 if bitstring[0] == '0' else -1
        #         z_exp += sign * count
        #     return z_exp / shots

        counts = job.get_counts()
        print("Shot counts after transpilation : ", job.get_counts())
        isa_circuit = qc  # added for consistency with function return for all backends (not an actual ISA circuit, ionq doesnt have those)
        return counts, isa_circuit

    elif backend_name in ibm_backends:
        # IBM hardware / fake IBM backend path:
        # transpile to a backend-compatible ISA circuit, then execute it directly.
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        isa_circuit = pm.run(qc)

        result = backend.run(isa_circuit, shots=shots).result()
        counts = result.get_counts()

        return counts, isa_circuit

    else:
        raise ValueError(
            f"Unsupported backend_name={backend_name}. "
            "Expected IBM, IonQ, or Aer backend."
        )