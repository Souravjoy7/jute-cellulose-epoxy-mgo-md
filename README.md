# Atomistic Reinforcement Mechanisms at the Jute Cellulose/Epoxy/Crystalline MgO Nanophase Interface: A Machine Learning Molecular Dynamics Study

This repository contains the complete simulation workflows, machine learning molecular dynamics (ML-MD) scripts, analysis pipelines, and post-processing tools used in the research paper.

## Repository Structure
- `models/`: Scripts and structural files for constructing Representative Volume Elements (RVEs) of Cellulose/Epoxy Control and MgO-reinforced systems.
- `simulations/`: Python workflows using ASE and MACE-MP neural network potential for NVT thermal equilibration (300 K, 350 K, 400 K) and Steered Molecular Dynamics (SMD) pull-out simulations.
- `analysis/`: Scripts for computing Kelly-Tyson Interfacial Shear Strength (ICSS), Work of Adhesion (Wad), Thermodynamic Interaction Energy (Eint), and Radial Distribution Functions (RDF).
- `data/`: Processed CSV output data files for all temperatures (300 K, 350 K, 400 K).
- `plotting/`: Publication-quality (600 DPI) visualization and curve generation scripts.

## Requirements
```bash
pip install ase mace-torch torch numpy scipy matplotlib pandas
```

## Citation
If you use these scripts or data in your research, please cite:
> Sree. Sourov Kumar, "Atomistic Reinforcement Mechanisms at the Jute Cellulose/Epoxy/Crystalline MgO Nanophase Interface: A Machine Learning Molecular Dynamics Study", *Computational Materials Science*, 2026.
