
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler


from evolve import evolve_and_measure_circuit

def meas_counts(t, pauli_terms, N_sites, theta_nu, trotter_steps, trotter_order, measure, backend_name, backend, shots):
    # Evolve and measure circuit based on the provided Pauli term (X, Y, Z)
    qc = evolve_and_measure_circuit(t, pauli_terms, N_sites, theta_nu, trotter_steps, trotter_order, measure=measure)

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
    
    return counts, isa_circuit
