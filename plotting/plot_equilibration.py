import os
import matplotlib.pyplot as plt
import numpy as np

def parse_eq_log(filepath):
    times = []
    pot_energies = []
    temps = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return None, None, None
        
    with open(filepath, "r") as f:
        header = f.readline()  # Read header
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    time = float(parts[0])
                    epot = float(parts[2])
                    temp = float(parts[4])
                    times.append(time)
                    pot_energies.append(epot)
                    temps.append(temp)
                except ValueError:
                    continue
                    
    return np.array(times), np.array(pot_energies), np.array(temps)

def main():
    # Load files
    data = {
        "Control (300 K)": parse_eq_log("control_eq.log"),
        "Control (400 K)": parse_eq_log("control_400K_eq.log"),
        "MgO Reinforced (350 K)": parse_eq_log("reinforced_350K_eq.log"),
        "MgO Reinforced (400 K)": parse_eq_log("reinforced_400K_eq.log")
    }
    
    # Setup styling
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = {
        "Control (300 K)": "#d9534f",
        "Control (400 K)": "#9b59b6",
        "MgO Reinforced (350 K)": "#5bc0de",
        "MgO Reinforced (400 K)": "#5cb85c"
    }
    
    styles = {
        "Control (300 K)": "--",
        "Control (400 K)": "--",
        "MgO Reinforced (350 K)": "-",
        "MgO Reinforced (400 K)": "-"
    }

    print("Equilibration Statistics:")
    print(f"{'System':<30} | {'Mean Temp (K)':<15} | {'Std Temp (K)':<12} | {'Mean Epot (eV)':<15}")
    print("-" * 80)
    
    for name, (times, epots, temps) in data.items():
        if times is None or len(times) == 0:
            continue
            
        color = colors[name]
        style = styles[name]
        
        # Plot Temperature vs Time
        ax1.plot(times, temps, label=name, color=color, linestyle=style, linewidth=2.0)
        
        # Plot Relative Potential Energy vs Time
        # Subtract initial potential energy to align them
        rel_epot = epots - epots[0]
        ax2.plot(times, rel_epot, label=name, color=color, linestyle=style, linewidth=2.0)
        
        # Calculate statistics (second half of equilibration)
        half_idx = len(temps) // 2
        mean_t = np.mean(temps[half_idx:])
        std_t = np.std(temps[half_idx:])
        mean_e = np.mean(epots[half_idx:])
        
        print(f"{name:<30} | {mean_t:14.2f} | {std_t:11.2f} | {mean_e:14.2f}")
        
    ax1.set_xlabel("Simulation Time (ps)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Temperature $T$ (K)", fontsize=12, fontweight="bold")
    ax1.set_title("NVT Equilibration Temperature Profile", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper right")
    
    ax2.set_xlabel("Simulation Time (ps)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Relative Potential Energy $\Delta E_{\\mathrm{pot}}$ (eV)", fontsize=12, fontweight="bold")
    ax2.set_title("NVT Equilibration Potential Energy Profile", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="upper right")
    
    plt.tight_layout()
    plot_path = "equilibration_proof.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nGenerated equilibration proof plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
