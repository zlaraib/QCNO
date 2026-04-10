
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
    qc = qc_base.copy()
    qc.save_statevector()
    qc.save_density_matrix()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuit = pm.run(qc)

    result = backend.run(isa_circuit).result()
    data = result.data()

    density_matrix = DensityMatrix(data["density_matrix"])
    statevector = Statevector(data["statevector"])

    print("Statevector retrieved successfully.")
    return isa_circuit, density_matrix, statevector

def meas_counts(qc_base, measure, N_sites, backend, backend_name, optimization_level, shots):
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
    if backend_name == "manila" or backend_name == "ibm"  or backend_name == "guadalupe": # decomposes all the 2 qubit gates in the circuit for 1 time step and optimizes all those 2-qubit gates as a whole for each time step. 

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
        
        # decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='ZXZ')
        decomposer = TwoQubitBasisDecomposer(ECRGate(), euler_basis='U3')
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

        # Transpile the new circuit to IBM native gates
        # transpiled_circuit = transpile(new_qc, basis_gates=['rz', 'sx','x', 'cx'], optimization_level=optimization_level)
        transpiled_circuit = transpile(new_qc,basis_gates=['rz','sx', 'x','cx', 'u3'], optimization_level=optimization_level)
        

        # Print the gate counts in the transpiled circuit
        gate_counts = transpiled_circuit.count_ops()
        print("\nGate counts in the transpiled circuit:")
        print(gate_counts)
        print("Circuit depth in transpiled circuit= ", transpiled_circuit.depth())
        qc = transpiled_circuit
    
    if backend_name == 'ionq' or backend_name =='ionq_noisy_sim' or backend_name =='ionq_native' or backend_name =='ionq_qpu':

        # Decompose two-qubit gates using the 'XYX' Euler basis for IonQ devices
        decomposer = TwoQubitBasisDecomposer(ECRGate(), euler_basis='XYX')

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

        # Transpile the circuit using IonQ's native gates: RX, RY, RXX, MS
        transpiled_circuit = transpile(new_qc, basis_gates=['rx', 'ry', 'rxx', 'ms'], optimization_level=optimization_level)
        qc = transpiled_circuit
        
        # # qc = transpile(qc, backend=backend, basis_gates=['rx', 'ry', 'rz', 'cx']) #added basis gateset for testing the noise model, can be removed for any case really 
        # qc = transpile(qc, backend=backend)
        # print(qc.draw())
        # gate counts in the transpiled circuit 
        gate_counts = qc.count_ops() 
        print("\nGate counts in the transpiled circuit:")
        print(gate_counts)
        print("Circuit depth in transpiled circuit= ", qc.depth())
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
    else:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        isa_circuit = pm.run(qc)

        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]

        counts = result[0].data.meas.get_counts()
        # result = backend.run(isa_circuit).result()
        return counts, isa_circuit
