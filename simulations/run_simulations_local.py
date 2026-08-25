import os
import sys
import time
import csv
import numpy as np
import torch
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import mace_mp

# Configure logging
def log(msg):
    print(f"[MD-PIPELINE] {msg}")
    sys.stdout.flush()

class SteeredCalculator:
    def __init__(self, base_calc, pulled_indices, spring_k, velocity, dt, initial_com_z):
        self.base_calc = base_calc
        self.pulled_indices = pulled_indices
        self.spring_k = spring_k      # eV/A^2
        self.velocity = velocity      # A/fs
        self.dt = dt                  # fs
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
        write(self.traj_file, self.atoms, append=True)
        self.atoms.calc = current_calc

def run_single_simulation(input_xyz, target_temp, output_dir, run_steps_override=None):
    log(f"==================================================")
    log(f" STARTING SIMULATION FOR T = {target_temp} K")
    log(f"==================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read atoms
    atoms = read(input_xyz)
    n_atoms = len(atoms)
    log(f"Loaded system: {n_atoms} atoms.")
    
    # Define groups
    # CNC bundle is first 1728 atoms
    cellulose_indices = list(range(0, 1728))
    
    # MgO sphere is next 360 atoms (indices 1728 to 2087)
    symbols = np.array(atoms.get_chemical_symbols())
    mg_indices = np.where(symbols == 'Mg')[0]
    
    if len(mg_indices) > 0:
        # Fixed MgO atoms
        fixed_indices = list(range(1728, 1728 + len(mg_indices) * 2))
        log(f"MgO Reinforced System: Fixed {len(fixed_indices)} MgO atoms (indices {fixed_indices[0]} to {fixed_indices[-1]}).")
    else:
        log("Error: No MgO atoms found in reinforced system input!")
        sys.exit(1)
        
    # Apply FixAtoms constraints
    constraint = FixAtoms(indices=fixed_indices)
    atoms.set_constraint(constraint)
    
    # Initialize MACE calculator on local GPU
    log("Initializing MACE-MP calculator on GPU...")
    base_calc = mace_mp(model="small", device="cuda")
    atoms.calc = base_calc
    
    # 1. Energy Minimization (LBFGS)
    log("Running energy minimization (LBFGS)...")
    opt_steps = 100
    if run_steps_override is not None:
        opt_steps = run_steps_override[0]
        
    opt = LBFGS(atoms, logfile=os.path.join(output_dir, "opt.log"))
    start_time = time.time()
    opt.run(fmax=0.1, steps=opt_steps)
    log(f"Energy minimization completed in {time.time() - start_time:.2f}s.")
    
    # 2. Thermal Equilibration (NVT Langevin)
    log(f"Running thermal equilibration at {target_temp} K...")
    dt_fs = 0.5
    eq_steps = 1000
    if run_steps_override is not None:
        eq_steps = run_steps_override[1]
        
    # Initialize velocities to Maxwell-Boltzmann distribution
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp)
    
    dyn = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=target_temp, friction=0.002, logfile=os.path.join(output_dir, "eq.log"))
    
    start_time = time.time()
    # Run equilibration in chunks to print progress
    steps_per_chunk = 100 if eq_steps >= 100 else eq_steps
    for step in range(0, eq_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, eq_steps - step)
        dyn.run(steps=chunk_steps)
        
        # Print stats
        pe = atoms.get_potential_energy()
        ke = atoms.get_kinetic_energy()
        temp_current = ke / (1.5 * n_atoms * units.kB)
        log(f"Equilibration Progress: Step {step + chunk_steps}/{eq_steps} | PotEng = {pe:.3f} eV | Temp = {temp_current:.1f} K")
        
    log(f"Thermal equilibration completed in {time.time() - start_time:.2f}s.")
    
    # 3. Steered MD Pull-out
    log("Setting up Steered MD pull-out simulation...")
    positions = atoms.get_positions()
    pulled_pos = positions[cellulose_indices]
    initial_com_z = np.mean(pulled_pos[:, 2])
    
    spring_k = 5.0
    velocity = 0.001  # A/fs
    pull_steps = 4000
    if run_steps_override is not None:
        pull_steps = run_steps_override[2]
        
    steered_calc = SteeredCalculator(
        base_calc=base_calc,
        pulled_indices=cellulose_indices,
        spring_k=spring_k,
        velocity=velocity,
        dt=dt_fs,
        initial_com_z=initial_com_z
    )
    atoms.calc = steered_calc
    
    traj_file = os.path.join(output_dir, "trajectory.xyz")
    # Remove trajectory file if it already exists
    if os.path.exists(traj_file):
        os.remove(traj_file)
        
    dyn_pull = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=target_temp, friction=0.002, logfile=os.path.join(output_dir, "pull.log"))
    
    # Attach trajectory writer observer
    traj_interval = 50 if pull_steps >= 50 else 1
    writer_obj = TrajectoryWriter(atoms, base_calc, traj_file)
    dyn_pull.attach(writer_obj, interval=traj_interval)
    
    start_time = time.time()
    # Run in chunks to print progress
    steps_per_chunk = 200 if pull_steps >= 200 else pull_steps
    for step in range(0, pull_steps, steps_per_chunk):
        chunk_steps = min(steps_per_chunk, pull_steps - step)
        dyn_pull.run(steps=chunk_steps)
        
        # Print stats
        pe = atoms.get_potential_energy()
        ke = atoms.get_kinetic_energy()
        temp_current = ke / (1.5 * n_atoms * units.kB)
        curr_force = steered_calc.pulled_forces[-1][2] if steered_calc.pulled_forces else 0.0
        log(f"Pull-out Progress: Step {step + chunk_steps}/{pull_steps} | PotEng = {pe:.3f} eV | Temp = {temp_current:.1f} K | Force = {curr_force:.3f} eV/A")
        
    log(f"Pull-out simulation completed in {time.time() - start_time:.2f}s.")
    
    # Save pull-out force data to CSV
    csv_path = os.path.join(output_dir, "pullout_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (fs)", "Displacement (A)", "Force (eV/A)"])
        writer.writerows(steered_calc.pulled_forces)
    log(f"Saved CSV data to: {csv_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in quick test mode (short steps)")
    args = parser.parse_args()
    
    input_file = "system_initial.xyz"
    if not os.path.exists(input_file):
        log(f"Error: {input_file} not found! Please run build_system.py first.")
        sys.exit(1)
        
    # Standard steps: relaxation=100, equilibration=1000, pullout=4000
    steps = None
    if args.test:
        log("!!! RUNNING IN QUICK TEST MODE (10 steps each) !!!")
        steps = (5, 10, 10)
        
    # Temperatures to run for reinforced system
    temperatures = [300.0, 350.0, 400.0]
    
    for T in temperatures:
        output_dir = f"reinforced_{int(T)}K"
        run_single_simulation(input_file, T, output_dir, run_steps_override=steps)
        
    log("=== ALL SIMULATIONS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
