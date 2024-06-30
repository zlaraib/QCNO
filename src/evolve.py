# evolve.py 

import numpy as np
from scipy.linalg import expm
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from momentum import momentum
from constants import hbar, c , eV, MeV, GeV, G_F, kB

def evolve_and_measure_circuit(time, H, N_sites):
    U = Operator(expm(-1j * H * time))
    qc = QuantumCircuit(N_sites, 1)
    half_N_sites = N_sites // 2
    qc.x(range(half_N_sites))
    qc.unitary(U, range(N_sites), label="exp(-iHt)")
    qc.measure(0, 0)
    # print(qc.decompose(reps=2)) #reps repeats the deocmpostion that many times. 
    return qc


def evolve_and_measure_qcircuit(time, H, N_sites, measure='Z'):
    U = Operator(expm(-1j * H * time* 1/hbar))
    
    qc = QuantumCircuit(N_sites, 1)
    half_N_sites = N_sites // 2
    # qc.x(range(half_N_sites))
    
    # Initialize the first half of the chain as up-spin (|0>)
    # Initialize the second half of the chain as down-spin (|1>)
    for i in range(half_N_sites, N_sites):
        qc.x(i)
        
    qc.unitary(U, range(N_sites), label="exp(-iHt)")
    
    # transform basis from Z (up/down or |0>/|1>) to X (equal super position of |up>/ |down> to get the x basis =|+>/|->) using hadamard on the first qubit(b/c we are measurng the first qubit only)
    # the measurement of counts in x basis is done later in main script to compute <sigma_x>
    if measure == 'X':
        qc.h(0)
        
    # prepare the quantum state by transforming basis from Z to Y using sdg on the first qubit(b/c we are measurng the first qubit only)
    # Similarly, the 'Sdg' gate (S-dagger) is applied, which corresponds to the inverse of the phase gate 'S'. This introduces a phase shift of - pi/2.
    # Then, a Hadamard gate 'H' is applied. This sequence of gates transforms from Z basis to Y basis. 
    
    # the measurement of counts in Y basis is done later in main script to compute <sigma_y>
    elif measure == 'Y':
        qc.sdg(0)
        qc.h(0)

    qc.measure(0, 0)
    return qc

