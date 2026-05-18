# QCNO Repository Architecture Flowchart

```mermaid
graph TD
    A["🐳 DOCKER SETUP<br/>(Infrastructure)"]
    
    A --> B["NVIDIA CUDA 12.1.1<br/>ubuntu22.04"]
    A --> C["BUILD TOOLS<br/>gfortran, build-essential<br/>libhdf5, OpenMPI<br/>LAPACK/BLAS, libpnetcdf"]
    A --> D["QUANTUM PACKAGES<br/>Qiskit, Qiskit-Aer<br/>IBM Provider, IonQ Provider<br/>Qiskit-Nature"]
    
    E["⚙️ JENKINS CI/CD PIPELINE"] 
    
    E --> E1["Stage 1: Prerequisites<br/>✓ mpicc -v<br/>✓ nvidia-smi<br/>✓ nvcc -V<br/>✓ git submodule<br/>✓ pip3 list"]
    
    E1 --> E2["Stage 2: Vacuum Oscillations<br/>main_vac_osc.ipynb → Python → Execute → PDF"]
    E1 --> E3["Stage 3: Rogerro 2021<br/>main_Rog.ipynb → Python → Execute → PDF"]
    E1 --> E4["Stage 4: Richers Homogeneous<br/>Homogenous_FFI_Richers.ipynb → Execute → PDF"]
    E1 --> E5["Stage 5: Richers Inhomogeneous<br/>Inhomogenous_FFI_Richers.ipynb → Execute → PDF"]
    E1 --> E6["Stage 6: Josh Depolarization<br/>Homo_Josh_noise_depolarization.ipynb → Execute → PDF"]
    
    F["📊 QUANTUM CIRCUIT GENERATION PIPELINE"]
    
    F --> F0["Physics Parameters Input<br/>N: occupancies, x: positions<br/>Δp: momentum spacing, L: size<br/>ω: frequencies, B: coefficients<br/>p: momentum, periodic BC"]
    
    F0 --> F1["src/constants.py<br/>ℏ, c, eV/MeV/GeV<br/>G_F, k_B"]
    
    F1 --> F2["src/hamiltonian.py<br/>Build Pauli Terms<br/>XX, YY, ZZ interactions<br/>Single-body X,Y,Z terms"]
    F1 --> F3["src/momentum.py<br/>Normalize momentum vectors"]
    F1 --> F4["src/geometric_func.py<br/>Geometric factors<br/>between particle pairs"]
    
    F2 --> F5["src/shape_func.py<br/>Spatial shape functions<br/>for interactions"]
    F2 --> F6["src/evolve.py<br/>TIME EVOLUTION<br/>Single-qubit: RX, RY, RZ<br/>Two-qubit: RXX, RYY, RZZ<br/>Custom & Cartan decomposition<br/>Trotterization 1st/2nd order<br/>SWAP gates & qubit permutation"]
    F2 --> F7["src/perturb.py<br/>Perturbative corrections<br/>Circuit layer adjustments"]
    
    F5 --> F8["QUANTUM CIRCUIT<br/>Parameterized Gates<br/>Trotter Steps<br/>Measurement Basis Rotation<br/>Classical Measurements"]
    F6 --> F8
    F7 --> F8
    
    G["🔬 CIRCUIT EXECUTION & MEASUREMENT"]
    
    F8 --> G
    G --> G0["add_measurement_to_circuit<br/>X/Y/Z basis rotation<br/>measure_all"]
    
    G0 --> G1["meas_counts<br/>Multi-Backend Dispatcher"]
    
    G1 --> G2["AER Backend Path<br/>1. Generate Pass Manager<br/>2. Transpile circuit<br/>3. Use SamplerV2<br/>4. Save statevector & density matrix<br/>5. Return results"]
    
    G1 --> G3["IonQ Backend Path<br/>1. Direct execution on cloud<br/>2. Gate set: rx, ry, rxx, ms<br/>3. Retrieve counts<br/>4. Return counts & circuit"]
    
    G1 --> G4["IBM Backend Path<br/>1. Two-Qubit decomposition KAK<br/>2. Euler basis: ZYZ, U3, etc<br/>3. Transpile with basis gates<br/>4. Execute on backend<br/>5. Return counts"]
    
    G2 --> H["src/sigma_statistics.py<br/>POST-PROCESSING"]
    G3 --> H
    G4 --> H
    
    H --> H1["Statistical Analysis<br/>σ_x, σ_y, σ_z expectations<br/>Correlation functions<br/>Oscillation probabilities<br/>Error quantification"]
    
    H1 --> H2["OUTPUT<br/>Probability distributions<br/>Observable values<br/>PDF visualizations"]
    
    I["📚 QISKIT EXAMPLES<br/>(Educational)"]
    I --> I1["Circuit Building<br/>circuit-construction<br/>circuit_analysis<br/>conditional_gate<br/>circuit_visualization"]
    I --> I2["Structural Examples<br/>01_electronic_structure<br/>02_vibrational_structure<br/>11_quadratic_hamiltonian<br/>12_deuteron_binding_energy"]
    I --> I3["Algorithms & Backends<br/>Getting_started<br/>backends_example<br/>Qiskit Simulators<br/>ibm_qiskit"]
    I --> I4["Advanced Simulations<br/>Ising_time_evolution<br/>heisenberg_H<br/>7_matrix_product_state<br/>entanglement_swapping"]
    
    J["🔧 BUILD_CIRCUIT<br/>(Custom Physics Notebooks)"]
    J --> J1["vac_osc_qc.ipynb<br/>Vacuum Oscillation QC"]
    J --> J2["self_int_exact.ipynb<br/>Exact Self-Interaction"]
    J --> J3["self_int_qc.ipynb<br/>QC Self-Interaction"]
    J --> J4["self_int_evol_from_ising.ipynb<br/>Ising Evolution"]
    
    K["📦 OUTPUT & ARTIFACTS"]
    H2 --> K
    K --> K1["PDF Visualizations<br/>Archived in Jenkins"]
    K --> K2["Measurement Data<br/>CSV/JSON Results"]
    K --> K3["Workspace Cleanup<br/>Remove temp files<br/>Archive results"]
    
    style A fill:#ff9999
    style E fill:#99ccff
    style F fill:#99ff99
    style G fill:#ffcc99
    style H fill:#cc99ff
    style I fill:#ffff99
    style J fill:#ff99cc
    style K fill:#99ffcc
```

## Architecture Overview

This flowchart shows the complete QCNO infrastructure organized in 8 major sections:

### 1. **Docker Setup (Red)** 🐳
- NVIDIA CUDA 12.1.1 with Ubuntu 22.04
- Build tools (gfortran, OpenMPI, LAPACK, libhdf5)
- Quantum computing packages (Qiskit, IBM Provider, IonQ Provider)

### 2. **Jenkins CI/CD Pipeline (Blue)** ⚙️
- Prerequisites validation (compilers, GPU, dependencies)
- 5 physics simulation stages:
  - Vacuum oscillations
  - Rogerro 2021 full Hamiltonian
  - Richers 2021 homogeneous MF
  - Richers 2021 inhomogeneous MF
  - Josh depolarization noise model

### 3. **Quantum Circuit Generation (Green)** 📊
- **Physics Parameters** → Input system parameters
- **Constants Module** → Physical constants (ℏ, c, G_F, k_B)
- **Hamiltonian Construction** → Pauli term builder
- **Supporting Modules** → Momentum, geometry, shape functions
- **Evolution Engine** → Time evolution via Trotterization
- **Circuit Output** → Parameterized quantum gates

### 4. **Circuit Execution (Orange)** 🔬
- **Measurement Setup** → Basis rotation (X/Y/Z)
- **Multi-Backend Dispatcher** → Route to appropriate backend
- **3 Execution Paths**:
  - **Aer**: Classical GPU simulation with statevector tracking
  - **IonQ**: Cloud quantum execution with native gates
  - **IBM**: Hardware with KAK decomposition and transpilation

### 5. **Post-Processing (Purple)** 
- Statistical analysis of measurement counts
- Extract observables (σ_x, σ_y, σ_z)
- Correlation functions and error quantification

### 6. **Qiskit Examples (Yellow)** 📚
Educational notebooks demonstrating circuit building, structural problems, algorithms, and advanced simulations

### 7. **Build Circuit (Pink)** 🔧
Custom physics implementations for vacuum oscillations, self-interactions, and Ising evolution

### 8. **Output & Artifacts (Cyan)** 📦
PDF visualizations, measurement data, and workspace cleanup

