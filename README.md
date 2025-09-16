# Physics-Aware ILU Preconditioner

A C++/OpenMP implementation of an incomplete LU (ILU) preconditioner that incorporates physical insights from quantum transport to accelerate iterative solvers (GMRES/BiCGSTAB) for electronic structure simulations.

## 🚀 Key Features
- **Physics-Aware Filtering:** Preserves matrix entries based on physical importance:
  - **Quantum Transport Pathways:** Identifies atoms involved in electron conduction.
  - **Surface & Contact Atoms:** Prioritizes atoms connected to simulated electrodes.
  - **Energy-Dependent Resonance:** Keeps entries crucial near the simulation energy (within a 2 eV window).
- **High Performance:** Parallelized with OpenMP for efficient computation on multi-core processors.
- **Robust Benchmarking:** Built-in framework to test convergence against standard methods (e.g., MKL) across multiple system sizes and energy ranges.
