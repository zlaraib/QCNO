
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

def meas_counts(t, pauli_terms, N_sites, theta_nu, trotter_steps, trotter_order, measure, backend_name, backend,optimization_level, shots):
    # Evolve and measure circuit based on the provided Pauli term (X, Y, Z)
    qc = evolve_and_measure_circuit(t, pauli_terms, backend_name,backend, optimization_level, N_sites, theta_nu, trotter_steps, trotter_order, measure=measure)
    print("\nOriginal Circuit:")
    print(qc.draw())
    # Print the gate counts in the original circuit
    gate_counts = qc.count_ops()
    print("\nGate counts in the original circuit:")
    print(gate_counts)
    print("Circuit depth= ", qc.depth())
    
    
    if backend_name == "manila" or backend_name == "ibm" :
    
        # Decompose two-qubit gates using KAK decomposition
        decomposer = TwoQubitBasisDecomposer(ECRGate(), euler_basis='U3')

        # Iterate through two-qubit gates in the circuit and decompose them
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

       # Print the decomposed circuit
        print("\nDecomposed Circuit:")
        print(new_qc.draw())
        # Print the gate counts in the decomposed circuit
        gate_counts = new_qc.count_ops()
        print("\nGate counts in the decomposed circuit:")
        print(gate_counts)
        print("Circuit depth in decomposed circuit= ", new_qc.depth())
        
        # Transpile the new circuit to IBM native gates
        transpiled_circuit = transpile(new_qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=optimization_level)
        
        print("\nTranspiled Circuit (IBM native gates):")
        print(transpiled_circuit.draw())

        # Print the gate counts in the transpiled circuit
        gate_counts = transpiled_circuit.count_ops()
        print("\nGate counts in the transpiled circuit:")
        print(gate_counts)
        print("Circuit depth in transpiled circuit= ", transpiled_circuit.depth())
        qc = transpiled_circuit

        
    if backend_name == 'ionq':
        qc = transpile(qc, backend=backend)
        job = backend.run(qc, shots=shots)
        counts = job.get_counts()
        isa_circuit = qc  # added for consistency with function return for all backends (not an actual ISA circuit, ionq doesnt have those)
    else:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)
        
        # Get counts from the result
        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
    
    # Print gate counts
    gate_counts = isa_circuit.count_ops()
    print(f"Gate counts at time step {t}: {gate_counts}")
    
    # Count the number of two-qubit gates
    dag = circuit_to_dag(isa_circuit)
    op_list = []
    for qdx in dag.collect_2q_runs():
        op_list.extend([x for x in qdx if x.op.num_qubits == 2])

    count_2q_ops = len(op_list)
    print(f"Number of two-qubit gates at time step {t}: {count_2q_ops}")
    
    # Get and print the size of the circuit
    circuit_size = isa_circuit.size()
    print(f"Size of the circuit at time step {t}: {circuit_size}")
    
    return counts, isa_circuit
