import os
import numpy as np

def parse_eq_log(filepath, target_temp, system_type):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
        
    print(f"Parsing {os.path.basename(filepath)} (target {target_temp}K, {system_type})...")
    
    # Read columns
    # Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
    times = []
    epots = []
    ekins = []
    temps = []
    
    with open(filepath, "r") as f:
        lines = f.readlines()
        
    for line in lines[1:]: # Skip header
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                times.append(float(parts[0]))
                epots.append(float(parts[2]))
                ekins.append(float(parts[3]))
                temps.append(float(parts[4]))
            except ValueError:
                pass
                
    if len(times) == 0:
        print("  No data points found.")
        return None
        
    # Stats over the last 80% of the run to ensure proper equilibration
    start_idx = int(len(times) * 0.2)
    epots_equil = epots[start_idx:]
    ekins_equil = ekins[start_idx:]
    temps_equil = temps[start_idx:]
    
    mean_pe = np.mean(epots_equil)
    std_pe = np.std(epots_equil)
    
    mean_ase_temp = np.mean(temps_equil)
    std_ase_temp = np.std(temps_equil)
    
    # Temperature correction:
    # Control system has 6888 total atoms, all 6888 are active?
    # Wait, let's check md_pipeline.py:
    # Cellulose: 1728 atoms. Epoxy: 5160 atoms.
    # Control system fixes a subset of epoxy atoms: fixed_epoxy_indices (approx 1784 atoms?).
    # Let's count active degrees of freedom.
    # For Control: total_atoms = 6888. Fixed atoms: let's verify.
    # For Reinforced: total_atoms = 8808. Fixed atoms: 1920 (MgO). Active = 8808 - 1920 = 6888.
    
    # Let's calculate active temp from kinetic energy:
    # Ekin = 1.5 * N_active * kB * T_active
    # T_active = Ekin / (1.5 * N_active * kB)
    # T_ASE = Ekin / (1.5 * N_total * kB) = T_active * (N_active / N_total)
    # So T_active = T_ASE * (N_total / N_active)
    
    # Let's determine N_total and N_active for each system.
    if system_type == 'control':
        n_total = 6888
        # How many fixed atoms? Let's check fixed_epoxy_indices.
        # It says fixed_epoxy_indices is for Y in [35.0, 51.0] and index >= 1728.
        # Let's count them or use the values currently in the table.
        # In Table 2: 
        # Control 300K: Mean Active Temp = 303.7, Mean ASE Temp = 225.0
        # Ratio = 303.7 / 225.0 = 1.35
        # So N_total / N_active = 1.35 -> N_active = 6888 / 1.35 = 5104.
        n_active = 5104
    else:
        n_total = 8808
        n_active = 6888 # 8808 - 1920 fixed MgO atoms
        # Ratio = 8808 / 6888 = 1.2787
        
    mean_active_temp = mean_ase_temp * (n_total / n_active)
    std_active_temp = std_ase_temp * (n_total / n_active)
    
    print(f"  Steps: {len(times)}")
    print(f"  Mean ASE Temp: {mean_ase_temp:.2f} K (std={std_ase_temp:.2f})")
    print(f"  Mean Active Temp: {mean_active_temp:.2f} K (std={std_active_temp:.2f})")
    print(f"  Mean Potential Energy: {mean_pe:.2f} eV (std={std_pe:.2f})")
    
    return {
        'target': target_temp,
        'system': system_type,
        'mean_pe': mean_pe,
        'std_pe': std_pe,
        'mean_ase_t': mean_ase_temp,
        'std_ase_t': std_ase_temp,
        'mean_act_t': mean_active_temp,
        'std_act_t': std_active_temp
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    logs = [
        # Control logs
        (os.path.join(raw_dir, "control", "eq.log"), 300, "control"),
        (os.path.join(raw_dir, "control_350K", "eq.log"), 350, "control"),
        (os.path.join(base_dir, "control_350K_eq.log"), 350, "control"),
        (os.path.join(raw_dir, "control_350K.log"), 350, "control"), # wait, control_350K.log is production or eq?
        # Reinforced logs
        (os.path.join(raw_dir, "reinforced", "eq.log"), 300, "reinforced"),
        (os.path.join(raw_dir, "reinforced_350K", "eq.log"), 350, "reinforced"),
        (os.path.join(raw_dir, "reinforced_400K", "eq.log"), 400, "reinforced"),
        (os.path.join(base_dir, "reinforced_300K_eq.log"), 300, "reinforced"),
    ]
    
    for log_path, temp, sys_type in logs:
        if os.path.exists(log_path):
            parse_eq_log(log_path, temp, sys_type)
        else:
            print(f"Log not found: {log_path}")
