## Usage and Workflow

This repository is intended as a companion computational toolbox for the numerical evaluation of transport parameters of porous microstructures within the Johnson–Champoux–Allard–Lafarge–Pride (JCALP) framework.  
All codes are developed and tested with **FEniCSx v0.10.0**.

### Repository structure

The repository is organized according to the type of porous microstructure and the corresponding representative volume element (RVE):
```
JCALPy/
├── Fibrous/
│ ├── Entire Cell/
│ └── Half Cell/
└── Foam/
├── 1_8 Cell/
└── 1_16 Cell/
```

Each subfolder contains:
- a CAD geometry file (`.step`)
- a mesh generation script
- Python solvers for Laplace, Poisson, and Stokes problems

### Geometry

The CAD geometries define representative microcells of fibrous and foam-like porous materials.  
All geometrical configurations are taken from and consistent with those reported in:

Zieliński T., Venegas R., Perrot C., Červenka M., Chevillotte F., Attenborough K. (2020). *Benchmarks for microstructure-based modelling of sound absorbing rigid-frame porous media*. [Journal of Sound and Vibration, DOI: 10.1016/j.jsv.2020.115441](https://doi.org/10.1016/j.jsv.2020.115441)


### Mesh generation

For each microcell, the finite element mesh is generated from the corresponding CAD geometry by running the mesh generation script provided in the same folder.  
Mesh generation **must be performed prior to executing any solver**.

### Numerical solvers and computed quantities

Each microcell folder includes the following solvers, implementing the full variational formulations and boundary conditions required for homogenization-based transport parameter evaluation:

- **Poisson.py**  
  Solves the Poisson problem and computes:
  - static thermal permeability  
  - static thermal tortuosity

- **Stokes.py**  
  Solves the Stokes flow problem and computes:
  - static viscous permeability  
  - static viscous tortuosity

- **Laplace.py**  
  Solves the Laplace problem and computes:
  - viscous characteristic length  
  - inertial tortuosity

All solvers include the implementation of appropriate boundary conditions and save the resolved fields for external visualization and post-processing.

### Computational workflow

1. Select the porous microstructure and RVE configuration of interest.
2. Generate the finite element mesh using the mesh script in the corresponding folder.
3. Run the required solvers (`Poisson.py`, `Stokes.py`, `Laplace.py`) on the generated mesh.
4. Post-process the saved solution fields to extract transport parameters.
5. Use the computed parameters in semi-phenomenological JCALP models for the prediction of dynamic acoustic properties of porous materials.
