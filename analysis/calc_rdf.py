import os
import sys
import numpy as np
from ase.io import read

def calculate_rdf_pairs(atoms, indices1, indices2, r_max=6.0, bin_width=0.05):
    """
    Calculate the RDF between group 1 and group 2 of atoms.
    """
    pos1 = atoms.positions[indices1]
    pos2 = atoms.positions[indices2]
    
    n1 = len(pos1)
    n2 = len(pos2)
    
    if n1 == 0 or n2 == 0:
        return np.arange(0, r_max, bin_width), np.zeros(int(r_max/bin_width))
        
    # Compute all pairwise distances between pos1 and pos2
    # Using cell periodic boundary conditions if possible, or simple Euclidean
    # Since our box is 60x60x80 A, and r_max is 6 A, simple Euclidean is fine
    # or we can use minimum image convention
    cell = atoms.cell
    use_pbc = cell is not None and cell.volume > 1.0
    
    distances = []
    for p1 in pos1:
        diffs = pos2 - p1
        if use_pbc:
            # Apply periodic boundary conditions (minimum image convention)
            # diffs = diffs - cell.round(diffs / cell_lengths)
            # A simple way for orthogonal cells:
            cell_diag = np.diagonal(cell)
            for i in range(3):
                if cell_diag[i] > 0:
                    diffs[:, i] = diffs[:, i] - np.round(diffs[:, i] / cell_diag[i]) * cell_diag[i]
                    
        dists = np.linalg.norm(diffs, axis=1)
        dists = dists[dists <= r_max]
        distances.extend(dists)
        
    distances = np.array(distances)
    
    # Calculate histogram
    bins = np.arange(0, r_max + bin_width, bin_width)
    hist, bin_edges = np.histogram(distances, bins=bins)
    
    # Normalize RDF
    # g(r) = hist(r) / (4 * pi * r^2 * dr * rho)
    # where rho = n2 / Volume (bulk density of group 2)
    vol = atoms.get_volume() if use_pbc else (60.0 * 60.0 * 80.0) # default volume
    rho = n2 / vol
    
    r = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    dr = bin_width
    
    # Shell volume: V_shell = 4/3 * pi * (r_out^3 - r_in^3)
    r_in = bin_edges[:-1]
    r_out = bin_edges[1:]
    shell_vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
    
    # g(r) = hist / (n1 * shell_vol * rho)
    g_r = hist / (n1 * shell_vol * rho)
    
    return r, g_r

def process_system(xyz_path, output_prefix):
    print(f"\nCalculating RDF for: {xyz_path}")
    try:
        atoms = read(xyz_path, index=0)
    except Exception as e:
        print(f"Error reading {xyz_path}: {e}")
        return
        
    n_atoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    
    # 1. Define groups
    # Cellulose is always first 1,728 atoms (0 to 1727)
    cell_o_indices = [i for i in range(1728) if symbols[i] == 'O']
    cell_h_indices = [i for i in range(1728) if symbols[i] == 'H']
    
    # Matrix starts at 1728.
    # If MgO is present, n_atoms is 8,808. MgO is 1,920 atoms (1728 to 3647).
    # If MgO is absent (control), n_atoms is 6,888. Matrix is just Epoxy.
    has_mgo = (n_atoms > 8000)
    
    if has_mgo:
        # MgO group (indices 1728 to 3647)
        mgo_mg_indices = [i for i in range(1728, 3648) if symbols[i] == 'Mg']
        mgo_o_indices = [i for i in range(1728, 3648) if symbols[i] == 'O']
        # Epoxy group (indices 3648 to end)
        epoxy_o_indices = [i for i in range(3648, n_atoms) if symbols[i] == 'O']
        epoxy_h_indices = [i for i in range(3648, n_atoms) if symbols[i] == 'H']
    else:
        # Epoxy group (indices 1728 to end)
        epoxy_o_indices = [i for i in range(1728, n_atoms) if symbols[i] == 'O']
        epoxy_h_indices = [i for i in range(1728, n_atoms) if symbols[i] == 'H']
        
    # 2. Calculate RDFs
    results = {}
    
    if has_mgo:
        # Cellulose O - MgO Mg (Coordinate covalent ionic bond check)
        r, g_r = calculate_rdf_pairs(atoms, cell_o_indices, mgo_mg_indices)
        results["cell_O_mgo_Mg"] = (r, g_r)
        
        # Cellulose H - MgO O (Hydrogen bond check)
        r, g_r = calculate_rdf_pairs(atoms, cell_h_indices, mgo_o_indices)
        results["cell_H_mgo_O"] = (r, g_r)
        
    # Cellulose H - Epoxy O (Hydrogen bond check)
    r, g_r = calculate_rdf_pairs(atoms, cell_h_indices, epoxy_o_indices)
    results["cell_H_epoxy_O"] = (r, g_r)
    
    # Save to CSV
    csv_path = f"{output_prefix}_rdf.csv"
    with open(csv_path, "w") as f:
        f.write("r_A")
        for key in results.keys():
            f.write(f",{key}")
        f.write("\n")
        
        # Write rows
        n_bins = len(r)
        for i in range(n_bins):
            f.write(f"{r[i]:.3f}")
            for key in results.keys():
                f.write(f",{results[key][1][i]:.4f}")
            f.write("\n")
            
    print(f"Saved RDF data to: {csv_path}")

def main():
    folders = {
        "control_300K": ("control/trajectory.xyz", "control_300K"),
        "reinforced_300K": ("reinforced_300K_clean/trajectory.xyz", "reinforced_300K"),
        "control_350K": ("control_350K/trajectory.xyz", "control_350K"),
        "reinforced_350K": ("reinforced_350K/trajectory.xyz", "reinforced_350K"),
        "control_400K": ("control_400K/trajectory.xyz", "control_400K"),
        "reinforced_400K": ("reinforced_400K/trajectory.xyz", "reinforced_400K")
    }
    
    for key, (path, prefix) in folders.items():
        if os.path.exists(path):
            process_system(path, prefix)
        else:
            print(f"Skipping {prefix} (trajectory.xyz not found at {path})")

if __name__ == "__main__":
    main()
