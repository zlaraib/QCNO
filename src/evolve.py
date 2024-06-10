# evolve.py 

import numpy as np
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

def evolve_and_measure_circuit(time, H, N_sites):
    U = Operator(expm(-1j * H * time))
    qc = QuantumCircuit(N_sites, 1)
    half_N_sites = N_sites // 2
    qc.x(range(half_N_sites))
    qc.unitary(U, range(N_sites), label="exp(-iHt)")
    qc.measure(0, 0)
    return qc