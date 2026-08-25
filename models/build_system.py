import os
import sys
import urllib.request
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import bulk

# Configure logging
def log(msg):
    print(f"[BUILDER] {msg}")
    sys.stdout.flush()

def make_mgo_spherical_nanoparticle(radius=9.5):
    log(f"Building spherical MgO nanoparticle with radius={radius} A...")
    mgo_unit = bulk('MgO', 'rocksalt', a=4.211, cubic=True)
    mgo_super = mgo_unit * (6, 6, 6)
    positions = mgo_super.get_positions()
    com = mgo_super.get_center_of_mass()
    positions -= com
    mgo_super.set_positions(positions)
    
    dists = np.linalg.norm(positions, axis=1)
    mask = dists <= radius
    indices = np.where(mask)[0]
    mgo_sphere = mgo_super[indices]
    
    symbols = np.array(mgo_sphere.get_chemical_symbols())
    mg_indices = np.where(symbols == 'Mg')[0]
    o_indices = np.where(symbols == 'O')[0]
    
    n_mg = len(mg_indices)
    n_o = len(o_indices)
    
    sphere_positions = mgo_sphere.get_positions()
    sphere_dists = np.linalg.norm(sphere_positions, axis=1)
    
    if n_mg > n_o:
        diff = n_mg - n_o
        mg_dists = sphere_dists[mg_indices]
        sorted_mg_sub_indices = np.argsort(mg_dists)[::-1]
        remove_indices = mg_indices[sorted_mg_sub_indices[:diff]]
        keep_mask = np.ones(len(mgo_sphere), dtype=bool)
        keep_mask[remove_indices] = False
        mgo_sphere = mgo_sphere[keep_mask]
    elif n_o > n_mg:
        diff = n_o - n_mg
        o_dists = sphere_dists[o_indices]
        sorted_o_sub_indices = np.argsort(o_dists)[::-1]
        remove_indices = o_indices[sorted_o_sub_indices[:diff]]
        keep_mask = np.ones(len(mgo_sphere), dtype=bool)
        keep_mask[remove_indices] = False
        mgo_sphere = mgo_sphere[keep_mask]
        
    log(f"Spherical MgO nanoparticle generated: Mg={sum(np.array(mgo_sphere.get_chemical_symbols()) == 'Mg')}, O={sum(np.array(mgo_sphere.get_chemical_symbols()) == 'O')}, Total={len(mgo_sphere)} atoms.")
    return mgo_sphere

def download_cellulose_cif(target_path="cellulose.cif"):
    # COD ID 4114994 is Cellulose I-beta (Nishiyama et al., 2002)
    url = "http://www.crystallography.net/cod/4114994.cif"
    log(f"Downloading Cellulose I-beta unit cell from COD: {url}")
    try:
        urllib.request.urlretrieve(url, target_path)
        log("Cellulose CIF downloaded successfully.")
    except Exception as e:
        log(f"Error downloading CIF: {e}")
        # Fallback coordinate definition if COD is down
        sys.exit(1)

def generate_molecules_from_smiles():
    log("Generating 3D structures for DGEBA and DETDA using RDKit...")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        log("Error: RDKit is required for structure generation from SMILES. Please install rdkit.")
        sys.exit(1)

    # 1. DGEBA (Bisphenol A Diglycidyl Ether) - C21 H24 O4
    dgeba_smiles = "CC(C)(c1ccc(OCC2CO2)cc1)c3ccc(OCC4CO4)cc3"
    mol_dgeba = Chem.MolFromSmiles(dgeba_smiles)
    mol_dgeba = Chem.AddHs(mol_dgeba)
    AllChem.EmbedMolecule(mol_dgeba, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_dgeba)
    conf_dgeba = mol_dgeba.GetConformer()
    dgeba_positions = conf_dgeba.GetPositions()
    dgeba_symbols = [atom.GetSymbol() for atom in mol_dgeba.GetAtoms()]
    dgeba_atoms = Atoms(symbols=dgeba_symbols, positions=dgeba_positions)
    log(f"Generated DGEBA molecule with {len(dgeba_atoms)} atoms.")

    # 2. DETDA (Diethyltoluenediamine - 2,4 isomer) - C11 H18 N2
    detda_smiles = "CCC1=C(C(=C(C(=C1)C)N)CC)N"
    mol_detda = Chem.MolFromSmiles(detda_smiles)
    mol_detda = Chem.AddHs(mol_detda)
    AllChem.EmbedMolecule(mol_detda, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_detda)
    conf_detda = mol_detda.GetConformer()
    detda_positions = conf_detda.GetPositions()
    detda_symbols = [atom.GetSymbol() for atom in mol_detda.GetAtoms()]
    detda_atoms = Atoms(symbols=detda_symbols, positions=detda_positions)
    log(f"Generated DETDA molecule with {len(detda_atoms)} atoms.")

    return dgeba_atoms, detda_atoms

def pack_matrix(box_size, base_atoms, dgeba_mol, detda_mol, n_dgeba=80, n_detda=40, min_dist=2.2):
    log(f"Packing {n_dgeba} DGEBA and {n_detda} DETDA molecules into the box...")
    
    # We will build a list of existing atom positions to check for overlaps
    existing_positions = base_atoms.get_positions()
    lx, ly, lz = box_size
    
    # Combined atoms container
    combined = base_atoms.copy()
    
    # Set seed for random packing
    np.random.seed(1234)
    
    packed_dgeba = 0
    packed_detda = 0
    
    # Pack DGEBA
    attempts = 0
    max_attempts = 1000
    while packed_dgeba < n_dgeba and attempts < max_attempts * n_dgeba:
        attempts += 1
        mol = dgeba_mol.copy()
        
        # Apply random rotation
        rot_axis = np.random.randn(3)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.random.uniform(0, 360)
        mol.rotate(rot_angle, rot_axis, center='COM')
        
        # Apply random translation inside the box
        pos = np.random.uniform([5, 5, 5], [lx-5, ly-5, lz-5])
        mol.translate(pos - mol.get_center_of_mass())
        
        # Check for overlaps with existing positions
        mol_pos = mol.get_positions()
        # Vectorized periodic distance check (minimum image convention)
        diff = mol_pos[:, np.newaxis, :] - existing_positions[np.newaxis, :, :]
        box = np.array(box_size)
        diff = diff - np.round(diff / box) * box
        dists = np.linalg.norm(diff, axis=2)
        if np.min(dists) >= min_dist:
            combined += mol
            existing_positions = np.concatenate([existing_positions, mol_pos], axis=0)
            packed_dgeba += 1
            if packed_dgeba % 20 == 0:
                log(f"Packed {packed_dgeba}/{n_dgeba} DGEBA molecules...")
                
    # Pack DETDA
    attempts = 0
    while packed_detda < n_detda and attempts < max_attempts * n_detda:
        attempts += 1
        mol = detda_mol.copy()
        
        # Apply random rotation
        rot_axis = np.random.randn(3)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.random.uniform(0, 360)
        mol.rotate(rot_angle, rot_axis, center='COM')
        
        # Apply random translation inside the box
        pos = np.random.uniform([5, 5, 5], [lx-5, ly-5, lz-5])
        mol.translate(pos - mol.get_center_of_mass())
        
        # Check for overlaps
        mol_pos = mol.get_positions()
        diff = mol_pos[:, np.newaxis, :] - existing_positions[np.newaxis, :, :]
        box = np.array(box_size)
        diff = diff - np.round(diff / box) * box
        dists = np.linalg.norm(diff, axis=2)
        if np.min(dists) >= min_dist:
            combined += mol
            existing_positions = np.concatenate([existing_positions, mol_pos], axis=0)
            packed_detda += 1
            if packed_detda % 10 == 0:
                log(f"Packed {packed_detda}/{n_detda} DETDA molecules...")
                
    log(f"Packing completed. DGEBA packed: {packed_dgeba}, DETDA packed: {packed_detda}")
    return combined

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-mgo", action="store_true", help="Exclude MgO slab to build control system")
    args = parser.parse_args()

    log("=== STARTING SYSTEM GENERATION ===")
    
    # 1. Download and load cellulose unit cell
    download_cellulose_cif("cellulose.cif")
    cellulose_unit = read("cellulose.cif")
    log(f"Cellulose unit cell loaded: {len(cellulose_unit)} atoms.")
    
    # 2. Build Cellulose Nanocrystal (CNC) bundle
    # We build a 2x2x6 supercell of cellobiose (8 chains, 6 unit cells along z-axis)
    # The chain direction is along c (z-axis)
    cnc = cellulose_unit * (2, 2, 6)
    log(f"Built Cellulose I-beta CNC bundle: {len(cnc)} atoms.")
    
    # Center the CNC bundle in the box
    # Let's define the simulation box size: 60 x 60 x 80 Å
    box_size = [60.0, 60.0, 80.0]
    
    # Shift CNC bundle to the center of the x-y plane, and bottom-aligned in z
    cnc.translate([30.0 - cnc.get_center_of_mass()[0], 22.0 - cnc.get_center_of_mass()[1], 40.0 - cnc.get_center_of_mass()[2]])
    
    if not args.exclude_mgo:
        # 3. Build MgO spherical nanoparticle
        mgo_sphere = make_mgo_spherical_nanoparticle(radius=9.5)
        
        # Center the MgO sphere COM at: X=30.0, Y=38.5, Z=40.0 (aligned with cellulose bundle in X/Z, zero-gap contact in Y)
        mgo_sphere.translate([30.0 - mgo_sphere.get_center_of_mass()[0], 38.5 - mgo_sphere.get_center_of_mass()[1], 40.0 - mgo_sphere.get_center_of_mass()[2]])
        
        # Combine Cellulose and MgO
        system = cnc + mgo_sphere
    else:
        log("Excluding MgO slab. Generating control system...")
        system = cnc
    
    # 4. Generate Epoxy monomers
    dgeba_mol, detda_mol = generate_molecules_from_smiles()
    
    # 5. Pack epoxy matrix around Cellulose and MgO
    full_system = pack_matrix(
        box_size=box_size,
        base_atoms=system,
        dgeba_mol=dgeba_mol,
        detda_mol=detda_mol,
        n_dgeba=80,
        n_detda=40,
        min_dist=2.2
    )
    
    # Set the cell and periodic boundary conditions
    full_system.set_cell(box_size)
    full_system.set_pbc([True, True, True])
    
    # 6. Save the structure in XYZ and PDB format
    output_prefix = "system_control" if args.exclude_mgo else "system_initial"
    write(f"{output_prefix}.xyz", full_system)
    write(f"{output_prefix}.pdb", full_system)
    log(f"Initial structure saved as {output_prefix}.xyz/.pdb. Total atoms in system: {len(full_system)}")
    log("=== SYSTEM GENERATION SUCCESSFUL ===")

if __name__ == "__main__":
    main()
