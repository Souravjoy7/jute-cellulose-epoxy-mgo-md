import os
import sys
import time
import numpy as np
import modal

# Define the Modal App
app = modal.App("matsci-sim-mace")

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
        "pandas",
        "matplotlib",
        "h5py",
    )
)

# Custom wrapper calculator to implement Steered Molecular Dynamics (SMD)
class SteeredCalculator:
    def __init__(self, base_calc, pulled_indices, spring_k, velocity, dt, initial_com_z):
        self.base_calc = base_calc
        self.pulled_indices = pulled_indices
        self.spring_k = spring_k      # in eV/A^2
        self.velocity = velocity      # in A/fs
        self.dt = dt                  # in fs
        self.initial_com_z = initial_com_z
        self.step = 0
        self.pulled_forces = []       # List to store (time_fs, displacement_A, force_eV_A)

    def get_forces(self, atoms):
        # 1. Get baseline forces from MACE calculator
        forces = self.base_calc.get_forces(atoms)
        
        # 2. Calculate current COM of pulled atoms (cellulose)
        positions = atoms.get_positions()
        pulled_pos = positions[self.pulled_indices]
        com_z = np.mean(pulled_pos[:, 2])
        
        # 3. Calculate target position of the spring
        target_z = self.initial_com_z + self.velocity * (self.step * self.dt)
        
        # 4. Calculate spring force along z-axis
        # F = k * (target_z - current_z)
        total_spring_force = self.spring_k * (target_z - com_z)
        
        # 5. Distribute the force among all pulled atoms
        n_pulled = len(self.pulled_indices)
        force_per_atom = total_spring_force / n_pulled
        forces[self.pulled_indices, 2] += force_per_atom
        
        # Record data
        displacement = target_z - self.initial_com_z
        self.pulled_forces.append((self.step * self.dt, displacement, total_spring_force))
        
        self.step += 1
        return forces

    def get_potential_energy(self, atoms, force_consistent=False):
        return self.base_calc.get_potential_energy(atoms, force_consistent)

    def get_stress(self, atoms):
        return self.base_calc.get_stress(atoms)


class TrajectoryWriter:
    def __init__(self, atoms, base_calc, traj_file):
        self.atoms = atoms
        self.base_calc = base_calc
        self.traj_file = traj_file

    def __call__(self):
        current_calc = self.atoms.calc
        self.atoms.calc = self.base_calc
        from ase.io import write
        write(self.traj_file, self.atoms, append=True)
        self.atoms.calc = current_calc


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=7200,  # 2 hours limit
    volumes={"/cache": volume}
)
def run_md_simulation(xyz_bytes: bytes, test_mode: bool = True, temperature: float = 300.0, run_name: str = "run") -> tuple:
    import numpy as np
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    from ase.md.langevin import Langevin
    from ase import units
    from mace.calculators import mace_mp
    
    # Configure HuggingFace cache to point to the persistent volume
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    checkpoint_file = f"/cache/checkpoint_{run_name}_{int(temperature)}K.traj"
    status_file = f"/cache/checkpoint_{run_name}_{int(temperature)}K.status"
    traj_file = f"/cache/trajectory_{run_name}_{int(temperature)}K.xyz"
    
    # Default initial states
    stage = "start"
    start_step = 0
    atoms = None
    
    # 1. Check for existing checkpoint to resume
    try:
        volume.reload()
    except Exception as e:
        print(f"Could not reload volume: {e}")
        
    if os.path.exists(status_file) and os.path.exists(checkpoint_file):
        try:
            with open(status_file, "r") as f:
                parts = f.read().strip().split(",")
                stage = parts[0]
                start_step = int(parts[1])
            atoms = read(checkpoint_file)
            print(f"Resuming simulation from stage '{stage}', step {start_step}.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            stage = "start"
            start_step = 0

    # If not resuming, load input structure from scratch
    if atoms is None:
        with open("input.xyz", "wb") as f:
            f.write(xyz_bytes)
        atoms = read("input.xyz")
        
    n_atoms = len(atoms)
    print(f"System contains {n_atoms} atoms.")
    
    # 2. Define groups
    # CNC bundle: first 1728 atoms
    cellulose_indices = list(range(0, 1728))
    
    # MgO slab: dynamically detect based on Mg atoms
    symbols = np.array(atoms.get_chemical_symbols())
    mg_indices = np.where(symbols == 'Mg')[0]
    
    if len(mg_indices) > 0:
        # Reinforced system: fix MgO slab
        n_mgo = 2 * len(mg_indices)
        fixed_indices = list(range(1728, 1728 + n_mgo))
        print(f"Reinforced system: Fixed {len(fixed_indices)} MgO atoms in place.")
    else:
        # Control system: fix Epoxy atoms in the Y-range [35.0, 51.0] to match the MgO boundary condition
        positions = atoms.get_positions()
        fixed_epoxy_indices = np.where((positions[:, 1] >= 35.0) & (positions[:, 1] <= 51.0) & (np.arange(len(atoms)) >= 1728))[0]
        fixed_indices = list(fixed_epoxy_indices)
        print(f"Control system: Fixed {len(fixed_indices)} Epoxy atoms in place (Y in [35.0, 51.0]) to match MgO boundary conditions.")
    
    # 3. Apply constraints
    constraint = FixAtoms(indices=fixed_indices)
    atoms.set_constraint(constraint)
    
    # 4. Initialize MACE-MP Calculator on GPU
    print("Loading MACE-MP foundation calculator on CUDA GPU...")
    # Use small model for speed and memory efficiency
    base_calc = mace_mp(model="small", device="cuda")
    atoms.calc = base_calc
    
    # 5. Energy Minimization (Relaxation)
    if stage == "start":
        print("Starting energy minimization (LBFGS)...")
        max_steps = 5 if test_mode else 100
        opt = LBFGS(atoms, logfile="opt.log")
        start_time = time.time()
        opt.run(fmax=0.1, steps=max_steps)
        print(f"Energy minimization completed in {time.time() - start_time:.2f}s.")
        
        # Save minimization checkpoint
        write(checkpoint_file, atoms)
        with open(status_file, "w") as f:
            f.write("equilibration,0")
        # volume.commit() removed to avoid slow cloud syncing
        stage = "equilibration"
        start_step = 0
    else:
        # Create dummy log if loading from checkpoint
        with open("opt.log", "w") as f:
            f.write("Energy minimization bypassed (resumed from checkpoint).\n")
            
    # 6. Thermal Equilibration (NVT Langevin Dynamics at 300K)
    dt_fs = 0.5  # 0.5 femtosecond timestep (safer for organic molecules with MACE-MP)
    temp = temperature # temperature in Kelvin
    eq_steps = 10 if test_mode else 1000  # 0.5 ps for production
    
    if stage == "equilibration":
        print(f"Starting/resuming thermal equilibration (Langevin) from step {start_step}...")
        if start_step == 0:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        dyn = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=temp, friction=0.002, logfile="eq.log")
        start_time = time.time()
        
        # Run in chunks of 50 steps to minimize Python loop overhead
        steps_per_chunk = 50
        for step in range(start_step, eq_steps, steps_per_chunk):
            chunk_steps = min(steps_per_chunk, eq_steps - step)
            print(f"Running equilibration steps {step} to {step + chunk_steps}...")
            dyn.run(steps=chunk_steps)
            time.sleep(0.1)  # Yield GIL and allow system events
            
            # Print temperature and energy progress
            pe = atoms.get_potential_energy()
            ke = atoms.get_kinetic_energy()
            temp_current = ke / (1.5 * len(atoms) * units.kB)
            print(f"Equilibration Progress: Step {step + chunk_steps}/{eq_steps} | PotEng = {pe:.3f} eV | Temp = {temp_current:.1f} K")
            
            # Checkpoint after the chunk
            write(checkpoint_file, atoms)
            with open(status_file, "w") as f:
                f.write(f"equilibration,{step + chunk_steps}")
            # volume.commit() removed to avoid slow cloud syncing
                    
        print(f"Thermal equilibration completed/resumed in {time.time() - start_time:.2f}s.")
        
        # Save checkpoint for pullout stage
        write(checkpoint_file, atoms)
        with open(status_file, "w") as f:
            f.write("pullout,0")
        # volume.commit() removed to avoid slow cloud syncing
        stage = "pullout"
        start_step = 0
    else:
        # Create dummy log if loading from checkpoint
        if not os.path.exists("eq.log"):
            with open("eq.log", "w") as f:
                f.write("Thermal equilibration bypassed (resumed from checkpoint).\n")
                
    # 7. Steered Molecular Dynamics (SMD) Pull-out Simulation
    print("Setting up Steered Molecular Dynamics (Pull-out)...")
    # Get initial center of mass of the cellulose CNC along the z-axis
    positions = atoms.get_positions()
    pulled_pos = positions[cellulose_indices]
    initial_com_z = np.mean(pulled_pos[:, 2])
    
    # Spring constant k = 5.0 eV/A^2
    spring_k = 5.0
    # Pulling velocity v = 0.001 A/fs (100 m/s)
    velocity = 0.001
    
    # Wrap the calculator with the Steered MD calculator
    steered_calc = SteeredCalculator(
        base_calc=base_calc,
        pulled_indices=cellulose_indices,
        spring_k=spring_k,
        velocity=velocity,
        dt=dt_fs,
        initial_com_z=initial_com_z
    )
    atoms.calc = steered_calc
    
    pull_steps = 10 if test_mode else 4000  # 4.0 ps for production
    
    if stage == "pullout":
        print(f"Running/resuming pull-out simulation for {pull_steps} steps from step {start_step}...")
        dyn_pull = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=temp, friction=0.002, logfile="pull.log")
        start_time = time.time()
        
        traj_interval = 2 if test_mode else 50
        
        # Set current step of steered calculator to start_step
        steered_calc.step = start_step
        
        # Attach trajectory writer observer
        writer_obj = TrajectoryWriter(atoms, base_calc, traj_file)
        dyn_pull.attach(writer_obj, interval=traj_interval)
        
        # Run in chunks of 50 steps
        steps_per_chunk = 50
        for step in range(start_step, pull_steps, steps_per_chunk):
            chunk_steps = min(steps_per_chunk, pull_steps - step)
            print(f"Running pull-out steps {step} to {step + chunk_steps}...")
            dyn_pull.run(steps=chunk_steps)
            time.sleep(0.1)  # Yield GIL and allow system events
            
            # Print temperature and energy progress
            pe = atoms.get_potential_energy()
            ke = atoms.get_kinetic_energy()
            temp_current = ke / (1.5 * len(atoms) * units.kB)
            # Find current force from steered calculator's log
            curr_force = steered_calc.pulled_forces[-1][2] if steered_calc.pulled_forces else 0.0
            print(f"Pull-out Progress: Step {step + chunk_steps}/{pull_steps} | PotEng = {pe:.3f} eV | Temp = {temp_current:.1f} K | Force = {curr_force:.3f} eV/A")
            
            # Checkpoint after the chunk
            write(checkpoint_file, atoms)
            with open(status_file, "w") as f:
                f.write(f"pullout,{step + chunk_steps}")
            # volume.commit() removed to avoid slow cloud syncing
                    
        print(f"Pull-out simulation completed/resumed in {time.time() - start_time:.2f}s.")
        
        # Clean up checkpoints on success
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            if os.path.exists(status_file):
                os.remove(status_file)
        except OSError:
            pass
            
    # Read output logs
    opt_log = ""
    eq_log = ""
    pull_log = ""
    if os.path.exists("opt.log"):
        with open("opt.log", "r") as f:
            opt_log = f.read()
    if os.path.exists("eq.log"):
        with open("eq.log", "r") as f:
            eq_log = f.read()
    if os.path.exists("pull.log"):
        with open("pull.log", "r") as f:
            pull_log = f.read()
            
    # Read trajectory bytes if created on the persistent volume
    traj_bytes = b""
    if os.path.exists(traj_file):
        with open(traj_file, "rb") as f:
            traj_bytes = f.read()
        # Clean up persistent trajectory file on successful run completion
        try:
            os.remove(traj_file)
        except OSError:
            pass
            
    # Pull-out force results
    forces_data = steered_calc.pulled_forces
    
    return forces_data, opt_log, eq_log, pull_log, traj_bytes


@app.local_entrypoint()
def main(input_xyz: str = "system_initial.xyz", test_mode: str = "true", output_dir: str = ".", temperature: float = 300.0, run_name: str = "run"):
    import csv
    import matplotlib.pyplot as plt
    
    is_test = test_mode.lower() == "true"
    print(f"Starting MD pipeline in {'TEST' if is_test else 'PRODUCTION'} mode.")
    
    if not os.path.exists(input_xyz):
        print(f"Error: Input file '{input_xyz}' not found.")
        sys.exit(1)
        
    with open(input_xyz, "rb") as f:
        xyz_bytes = f.read()
        
    print("Submitting Molecular Dynamics job to Modal GPU cluster...")
    start_time = time.time()
    forces_data, opt_log, eq_log, pull_log, traj_bytes = run_md_simulation.remote(xyz_bytes, is_test, temperature, run_name)
    print(f"Modal execution finished in {time.time() - start_time:.2f} seconds.")
    
    # Save the logs and results locally on VDS
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "opt.log"), "w") as f:
        f.write(opt_log)
    with open(os.path.join(output_dir, "eq.log"), "w") as f:
        f.write(eq_log)
    with open(os.path.join(output_dir, "pull.log"), "w") as f:
        f.write(pull_log)
        
    if traj_bytes:
        with open(os.path.join(output_dir, "trajectory.xyz"), "wb") as f:
            f.write(traj_bytes)
            
    # Save the force-displacement data to CSV
    csv_path = os.path.join(output_dir, "pullout_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (fs)", "Displacement (A)", "Force (eV/A)"])
        writer.writerows(forces_data)
        
    print(f"Saved CSV data to: {csv_path}")
    
    # Generate and save the pull-out force plot
    times, displacements, forces = zip(*forces_data)
    
    # Convert force from eV/A to NanoNewton (nN)
    # 1 eV/A = 1.602176634 nN
    forces_nN = [f * 1.602176634 for f in forces]
    
    plt.figure(figsize=(8, 5))
    plt.plot(displacements, forces_nN, label="Pull-out Force", color="navy", linewidth=2)
    plt.xlabel("Displacement ($\AA$)", fontsize=12)
    plt.ylabel("Pull-out Force (nN)", fontsize=12)
    plt.title("Steered MD Pull-out Test (Jute Cellulose / Epoxy / MgO)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(output_dir, "pullout_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated and saved publication-quality plot to: {plot_path}")
    
    # Print the peak force (Interfacial Shear Strength proxy)
    peak_force = max(forces_nN)
    print("\n" + "="*50)
    print(" SIMULATION SUCCESSFUL")
    print("="*50)
    print(f"Peak Pull-out Force: {peak_force:.3f} nN")
    print(f"Output files saved in: {output_dir}")
    print("="*50)
