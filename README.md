# Atomistic Reinforcement Mechanisms at the Jute Cellulose/Epoxy/Crystalline MgO Nanophase Interface: A Machine Learning Molecular Dynamics Study

This repository contains the complete simulation workflows, machine learning molecular dynamics (ML-MD) scripts, analysis pipelines, 3D atomic structures, and publication figures used in the research paper.

## Repository Structure
- `models/`: Scripts (`build_system.py`) and atomic coordinate files (`.xyz`, `.pdb`, `.cif`) for Cellulose, Epoxy, and MgO-reinforced RVEs.
- `simulations/`: Python workflows using ASE and MACE-MP neural network potential for NVT thermal equilibration (300 K, 350 K, 400 K) and Steered Molecular Dynamics (SMD) pull-out simulations.
- `analysis/`: Scripts for computing Kelly-Tyson Interfacial Shear Strength (ICSS), Work of Adhesion (Wad), Thermodynamic Interaction Energy (Eint), and Radial Distribution Functions (RDF).
- `data/`: Processed CSV output data files (pull-out force-displacement trajectories, RDFs, and summary tables) for all thermal regimes.
- `figures/`: High-resolution publication-ready figures (Figures 1 to 9, pull-out trajectories, RDFs, and correlation schematics).
- `plotting/`: Standalone Python visualization scripts for generating all figures.

## Requirements
```bash
pip install ase mace-torch torch numpy scipy matplotlib pandas
```

## Citation
If you use these scripts, coordinate structures, or datasets in your research, please cite:
> Sree. Sourov Kumar, "Atomistic Reinforcement Mechanisms at the Jute Cellulose/Epoxy/Crystalline MgO Nanophase Interface: A Machine Learning Molecular Dynamics Study", *Computational Materials Science*, 2026.
