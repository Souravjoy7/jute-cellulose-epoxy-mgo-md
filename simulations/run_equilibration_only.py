import os
import sys
import time
import modal

# Define the Modal App
app = modal.App("matsci-eq-only")

# Mount a persistent volume to cache downloaded MACE models
volume = modal.Volume.from_name("hf-models-cache", create_if_missing=True)

# Build a container image with PyTorch and MACE
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2.0.0",
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
    timeout=1200,
    volumes={"/cache": volume}
)
def run_eq_on_gpu(xyz_bytes: bytes) -> bytes:
    import torch
    import os
    from ase.io import read, write
    from ase.optimize import LBFGS
    from ase.md.langevin import Langevin
    from ase import units
    from mace.calculators import mace_mp
    
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    with open("input.xyz", "wb") as f:
        f.write(xyz_bytes)
        
    atoms = read("input.xyz")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="small", device=device, default_dtype="float32")
    atoms.calc = calc
    
    # 1. Minimization
    print("Running minimization...")
    opt = LBFGS(atoms, logfile="opt.log")
    opt.run(fmax=0.1, steps=100)
    
    # 2. Equilibration (1000 steps at 300K)
    print("Running equilibration (300K)...")
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
    dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300.0, friction=0.002, logfile="eq.log")
    
    # Run 1000 steps
    dyn.run(steps=1000)
    
    # Save the final frame as XYZ bytes
    write("equilibrated.xyz", atoms)
    with open("equilibrated.xyz", "rb") as f:
        eq_bytes = f.read()
        
    return eq_bytes

@app.local_entrypoint()
def main():
    if not os.path.exists("system_initial.xyz"):
        print("Error: system_initial.xyz not found.")
        sys.exit(1)
        
    with open("system_initial.xyz", "rb") as f:
        xyz_bytes = f.read()
        
    print("Submitting 300K Reinforced equilibration job to Modal GPU cluster...")
    eq_bytes = run_eq_on_gpu.remote(xyz_bytes)
    
    # Save it in a new folder to avoid conflicts
    os.makedirs("reinforced_300K_clean", exist_ok=True)
    with open("reinforced_300K_clean/trajectory.xyz", "wb") as f:
        f.write(eq_bytes)
        
    print("Successfully regenerated 300K Reinforced equilibrated structure at: reinforced_300K_clean/trajectory.xyz")

if __name__ == "__main__":
    main()
