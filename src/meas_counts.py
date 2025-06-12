
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import ECRGate, IGate, RZGate, SXGate, XGate, CXGate
from qiskit import transpile 
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from evolve import evolve_and_measure_circuit
from qiskit.quantum_info import Statevector,DensityMatrix
from qiskit.transpiler.passes import DynamicalDecoupling, ALAPSchedule
from qiskit.transpiler import PassManager
from qiskit.transpiler import InstructionDurations
import numpy as np
# from mitiq import zne
# from mitiq.zne.scaling import fold_gates_from_left
# from mitiq.zne.inference import RichardsonFactory

def meas_counts(t,times, N, omega, B, N_sites, Δx,  p,theta_nu, trotter_steps, trotter_order, measure, backend_name, backend,optimization_level, shots):

    # Evolve and measure circuit based on the provided Pauli term (X, Y, Z)
    qc = evolve_and_measure_circuit(t,  backend_name,backend,optimization_level, N, omega, B, N_sites, Δx,  p,theta_nu, trotter_steps, trotter_order, measure=measure)
    # for step in range(len(times)):
    #     for i in range(N_sites - 1):
    #         qc.cx(i, i + 1)
        
        # if step > 0:  # Apply SWAP after the first time step
        #     for i in range(0, N_sites - 1, 2):
        #         qc.swap(i, i + 1)
        
    print("\nOriginal Circuit:")
    # print(qc.draw())
    # Print the gate counts in the original circuit
    gate_counts = qc.count_ops()
    # print("\nGate counts in the original circuit:")
    print(gate_counts)
    # # Circuit depth = The maximum number of gates (or operations) applied sequentially on any qubit in the circuit. This is equivalent to the longest "path" of operations in the circuit.
    # # Depth provides a measure of how many time steps are required to execute the circuit i.e. layers of gates reuired in the circuit to execute from start to finish.
    print("Circuit depth= ", qc.depth())

    if backend_name == "manila" or backend_name == "ibm" : # decomposes all the 2 qubit gates in the circuit for 1 time step and optimizes all those 2-qubit gates as a whole for each time step. 

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
        # # Iterate through two-qubit gates in the circuit and decompose them
        new_qc = QuantumCircuit(N_sites, 1)
        for gate, qargs, cargs in qc.data:
            if gate.num_qubits == 2:
                # Get the unitary representation of the gate
                unitary_matrix = Operator(gate).data
                # Decompose using the TwoQubitBasisDecomposer
                decomposed_circuit = decomposer(unitary_matrix)
                # Append the decomposed gates to the new circuit
                new_qc.append(decomposed_circuit, qargs)
            else:
                # Directly append single-qubit gates or measurements to the new circuit
                new_qc.append(gate, qargs, cargs)

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
        
        # trying dynamical decoupling 
        # # Get backend properties
        # properties = backend.properties()

        # # Manually extract durations
        # durations = InstructionDurations.from_backend(backend)

        # # Step 1: Transpile the circuit
        # transpiled_circuit = transpile(
        #     new_qc,
        #     backend=backend,
        #     basis_gates=['rz', 'sx', 'x', 'cx', 'u3'],
        #     optimization_level=optimization_level
        # )

        # # Step 2: Schedule the circuit
        # schedule_pass = ALAPSchedule(durations=durations)
        # scheduled_pm = PassManager(schedule_pass)
        # scheduled_circuit = scheduled_pm.run(transpiled_circuit)

        # # Step 3: Add dynamical decoupling
        # dd_sequence = [XGate(), XGate()]
        # dd_pass = DynamicalDecoupling(durations, dd_sequence)
        # dd_pm = PassManager(dd_pass)
        # dd_circuit = dd_pm.run(scheduled_circuit)

        # qc = dd_circuit  # final circuit with DD

        # print("\nTranspiled Circuit (IBM native gates):")
        # print(transpiled_circuit.draw())

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
        new_qc = QuantumCircuit(N_sites, 1)

        for gate, qargs, cargs in qc.data:
            if gate.num_qubits == 2:
                # Get the unitary representation of the gate
                unitary_matrix = Operator(gate).data
                # Decompose using the TwoQubitBasisDecomposer
                decomposed_circuit = decomposer(unitary_matrix)
                # Append the decomposed gates to the new circuit
                new_qc.append(decomposed_circuit, qargs)
            elif gate.name == 'x':  # Replace X with RX(pi) for IonQ
                new_qc.rx(np.pi, qargs[0])  # RX(pi) is equivalent to X for IonQ
            else:
                # Directly append single-qubit gates or measurements to the new circuit
                new_qc.append(gate, qargs, cargs)

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
        def execute_circuit(circuit):
            # Submit job to your backend
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Optional: compute observable (example: parity or Z expectation)
            # Here's a simple observable: expectation value of Z on qubit 0
            z_exp = 0
            for bitstring, count in counts.items():
                sign = 1 if bitstring[0] == '0' else -1
                z_exp += sign * count
            return z_exp / shots

        # # Now apply ZNE
        # factory = RichardsonFactory(scale_factors=[1.0, 1.5, 2.0])
        # mitigated_result = zne.execute_with_zne(
        #     qc, 
        #     execute_circuit, 
        #     factory=factory,
        #     folding=fold_gates_from_left
        # )

        # print("ZNE-mitigated observable result:", mitigated_result)
        # print(job.job_id())
        counts = job.get_counts()
        print("Shot counts after transpilation : ", job.get_counts())
        isa_circuit = qc  # added for consistency with function return for all backends (not an actual ISA circuit, ionq doesnt have those)
    else:
        if backend_name == 'aer':
            # Add the save_density_matrix instruction
            qc.save_density_matrix()
            
        pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        isa_circuit = pm.run(qc)
        # Get counts from the result
        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        print("Shot counts after transpilation : ", counts)
        result = backend.run(isa_circuit).result()
        if backend_name == 'aer':
            # Retrieve the density matrix
            density_matrix = DensityMatrix(result.data()["density_matrix"])
            print("Density Matrix:")
            print(density_matrix)


    # # Print gate counts= Total number of gates.
    # gate_counts = isa_circuit.count_ops()
    # print(f"Gate counts at time step {t}: {gate_counts}") 
    
    # # Count the number of two-qubit gates
    # dag = circuit_to_dag(isa_circuit)
    # op_list = []
    # for qdx in dag.collect_2q_runs():
    #     op_list.extend([x for x in qdx if x.op.num_qubits == 2])

    # count_2q_ops = len(op_list)
    # print(f"Number of two-qubit gates at time step {t}: {count_2q_ops}")
    
    # # Get and print the size of the circuit =  
    # circuit_size = isa_circuit.size()
    # print(f"Size of the circuit at time step {t}: {circuit_size}")
    if backend_name == 'aer':
        return counts, isa_circuit, density_matrix
    else:
        return counts, isa_circuit
