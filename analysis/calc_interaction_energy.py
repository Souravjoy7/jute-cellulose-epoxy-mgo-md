import os
import sys
import time
import modal

# Define the Modal App
app = modal.App("matsci-calc-energy")

# Mount a persistent volume to cache downloaded MACE models
volume = modal.Volume.from_name("hf-models-cache", create_if_missing=True)

# Build a container image with PyTorch and MACE
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2.0.0",  # Maintain compatibility with matscipy
        "scipy",
        "ase",
        "mace-torch",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=600,
    volumes={"/cache": volume}
)
def compute_energy_on_gpu(trajectory_bytes: bytes) -> tuple:
    import torch
    import os
    from ase.io import read
    from mace.calculators import mace_mp
    
    # Configure HuggingFace cache to point to the persistent volume
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    with open("temp.xyz", "wb") as f:
        f.write(trajectory_bytes)
        
    # Read the first frame (fully equilibrated structure before pull-out)
    atoms = read("temp.xyz", index=0)
    n_atoms = len(atoms)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="small", device=device, default_dtype="float32")
    
    # 1. Total energy
    atoms.calc = calc
    e_total = atoms.get_potential_energy()
    
    # 2. Cellulose energy (first 1,728 atoms)
    cellulose_atoms = atoms[list(range(0, 1728))]
    cellulose_atoms.calc = calc
    e_cellulose = cellulose_atoms.get_potential_energy()
    
    # 3. Matrix energy (remaining atoms)
    matrix_atoms = atoms[list(range(1728, n_atoms))]
    matrix_atoms.calc = calc
    e_matrix = matrix_atoms.get_potential_energy()
    
    # 4. Interaction energy
    e_int = e_total - (e_cellulose + e_matrix)
    
    # Clean up temp file
    try:
        os.remove("temp.xyz")
    except OSError:
        pass
        
    return float(e_total), float(e_cellulose), float(e_matrix), float(e_int)

@app.local_entrypoint()
def main():
    folders = {
        "control_300K": ("control/trajectory.xyz", "Pure Epoxy Control (300K)"),
        "reinforced_300K": ("reinforced_300K_clean/trajectory.xyz", "MgO Reinforced (300K)"),
        "control_350K": ("control_350K/trajectory.xyz", "Pure Epoxy Control (350K)"),
        "reinforced_350K": ("reinforced_350K/trajectory.xyz", "MgO Reinforced (350K)"),
        "control_400K": ("control_400K/trajectory.xyz", "Pure Epoxy Control (400K)"),
        "reinforced_400K": ("reinforced_400K/trajectory.xyz", "MgO Reinforced (400K)")
    }
    
    results = []
    for key, (path, name) in folders.items():
        if os.path.exists(path):
            print(f"\nReading {path}...")
            with open(path, "rb") as f:
                traj_bytes = f.read()
            print(f"Submitting {name} to Modal GPU...")
            e_total, e_cellulose, e_matrix, e_int = compute_energy_on_gpu.remote(traj_bytes)
            print(f"Results for {name}:")
            print(f"  E_total     = {e_total:.3f} eV")
            print(f"  E_cellulose = {e_cellulose:.3f} eV")
            print(f"  E_matrix    = {e_matrix:.3f} eV")
            print(f"  E_int       = {e_int:.3f} eV")
            results.append({
                "name": name,
                "e_total": e_total,
                "e_cellulose": e_cellulose,
                "e_matrix": e_matrix,
                "e_int": e_int
            })
        else:
            print(f"Skipping {name} (trajectory.xyz not found at {path})")
            
    if results:
        import csv
        with open("interaction_energy_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["System", "E_total (eV)", "E_cellulose (eV)", "E_matrix (eV)", "Interaction Energy E_int (eV)", "E_int (kJ/mol)"])
            for r in results:
                # 1 eV = 96.4853 kJ/mol
                e_int_kj = r["e_int"] * 96.4853
                writer.writerow([r["name"], f"{r['e_total']:.3f}", f"{r['e_cellulose']:.3f}", f"{r['e_matrix']:.3f}", f"{r['e_int']:.3f}", f"{e_int_kj:.3f}"])
        print("\nSaved interaction energy summary to: interaction_energy_summary.csv")

if __name__ == "__main__":
    main()
