import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def process_simulation(csv_path, output_dir, name, D=16.4, L=62.3):
    """
    D: Equivalent diameter of CNC bundle in Angstroms
    L: Embedded length of CNC bundle in Angstroms
    """
    times = []
    displacements = []
    forces_ev_A = []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            times.append(float(row[0]))
            displacements.append(float(row[1]))
            forces_ev_A.append(float(row[2]))
            
    # Convert force to NanoNewton (nN)
    # 1 eV/A = 1.602176634 nN
    forces_nN = [f * 1.602176634 for f in forces_ev_A]
    
    # 1. Peak Force
    peak_force_nN = max(forces_nN)
    peak_idx = np.argmax(forces_nN)
    peak_disp = displacements[peak_idx]
    
    # 2. Interfacial Shear Strength (ICSS) in MPa
    # Area in m^2 = pi * D * L * 1e-20
    # Force in N = peak_force * 1e-9
    # ICSS in Pa = Force / Area
    # ICSS in MPa = ICSS * 1e-6
    area_m2 = np.pi * (D * 1e-10) * (L * 1e-10)
    peak_force_N = peak_force_nN * 1e-9
    icss_Pa = peak_force_N / area_m2
    icss_MPa = icss_Pa * 1e-6
    
    # 3. Work of Adhesion (Joule)
    # Integrating Force (nN) vs Displacement (Angstrom)
    # 1 nN * 1 Angstrom = 1e-9 N * 1e-10 m = 1e-19 Joules
    # Let's calculate the integral using Simpson's rule
    work_nN_A = simpson(y=forces_nN, x=displacements)
    work_J = work_nN_A * 1e-19
    work_eV = work_nN_A * 0.1 # 1 nN * A = 0.1 eV (since 1.602 nN * A = 1 eV, so 1 nN * A = 0.624 eV)
    # Wait: 1 eV = 1.602 nN * A, so work_eV = work_nN_A / 1.602176634
    work_eV = work_nN_A / 1.602176634
    
    # 4. Interfacial Stiffness (nN/A)
    # Find the slope in the initial linear region (first 10% of displacements or up to 1.0 A)
    linear_indices = [i for i, disp in enumerate(displacements) if disp <= 1.0]
    if len(linear_indices) >= 2:
        x_linear = [displacements[i] for i in linear_indices]
        y_linear = [forces_nN[i] for i in linear_indices]
        stiffness, _ = np.polyfit(x_linear, y_linear, 1)
    else:
        stiffness = 0.0
        
    print(f"--- RESULTS FOR {name} ---")
    print(f"Peak Pull-out Force: {peak_force_nN:.3f} nN (at displacement {peak_disp:.2f} Å)")
    print(f"Interfacial Shear Strength (ICSS): {icss_MPa:.2f} MPa")
    print(f"Work of Adhesion: {work_eV:.2f} eV ({work_J * 1e18:.3f} aJ)")
    print(f"Interfacial Stiffness: {stiffness:.3f} nN/Å")
    
    return {
        "name": name,
        "displacements": displacements,
        "forces_nN": forces_nN,
        "peak_force_nN": peak_force_nN,
        "peak_disp": peak_disp,
        "icss_MPa": icss_MPa,
        "work_eV": work_eV,
        "work_J": work_J,
        "stiffness_nN_A": stiffness
    }

def main():
    temps = [300, 350, 400]
    systems = ["control", "reinforced"]
    labels = {
        "control": "Pure Epoxy Control",
        "reinforced": "MgO Reinforced"
    }
    
    results = {}
    
    for t in temps:
        results[t] = {}
        for s in systems:
            folder = f"{s}_{t}K" if t != 300 else s
            csv_path = os.path.join(folder, "pullout_data.csv")
            if os.path.exists(csv_path):
                name = f"{labels[s]} ({t}K)"
                results[t][s] = process_simulation(csv_path, folder, name)
            else:
                print(f"Skipping {folder} (pullout_data.csv not found)")
                
    # Generate comparative plots for each temperature if both exist
    for t in temps:
        if "control" in results[t] and "reinforced" in results[t]:
            c = results[t]["control"]
            r = results[t]["reinforced"]
            
            plt.figure(figsize=(9, 5.5))
            plt.plot(c["displacements"], c["forces_nN"], label=f"Pure Epoxy (Control) - Peak: {c['peak_force_nN']:.2f} nN", color="#d9534f", linewidth=2.5, linestyle="--")
            plt.plot(r["displacements"], r["forces_nN"], label=f"MgO Reinforced - Peak: {r['peak_force_nN']:.2f} nN", color="#0275d8", linewidth=2.5)
            plt.xlabel("Displacement ($\AA$)", fontsize=12, fontweight="bold")
            plt.ylabel("Pull-out Force (nN)", fontsize=12, fontweight="bold")
            plt.title(f"Molecular Dynamics Pull-out Test Comparison ({t} K)", fontsize=13, fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=11, loc="upper right")
            
            plot_path = f"pullout_comparison_{t}K.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Generated comparative plot saved to: {plot_path}")
            
    # Generate Temperature vs ICSS plot if we have data for multiple temperatures
    available_temps = [t for t in temps if "control" in results[t] and "reinforced" in results[t]]
    if len(available_temps) >= 2:
        control_icss = [results[t]["control"]["icss_MPa"] for t in available_temps]
        reinforced_icss = [results[t]["reinforced"]["icss_MPa"] for t in available_temps]
        
        plt.figure(figsize=(8, 5.5))
        plt.plot(available_temps, control_icss, marker="o", markersize=8, label="Pure Epoxy Control", color="#d9534f", linewidth=3, linestyle="--")
        plt.plot(available_temps, reinforced_icss, marker="s", markersize=8, label="MgO Reinforced", color="#0275d8", linewidth=3)
        plt.xlabel("Temperature (K)", fontsize=12, fontweight="bold")
        plt.ylabel("Interfacial Shear Strength (ICSS, MPa)", fontsize=12, fontweight="bold")
        plt.title("Interfacial Shear Strength vs. Temperature", fontsize=13, fontweight="bold")
        plt.xticks(temps)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11, loc="lower left")
        
        plot_path = "icss_vs_temperature.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated Temperature vs ICSS plot saved to: {plot_path}")
        
        # Save comparison results to CSV
        with open("comparison_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Temperature (K)", "System", "Peak Force (nN)", "ICSS (MPa)", "Work of Adhesion (eV)", "Stiffness (nN/A)"])
            for t in available_temps:
                c = results[t]["control"]
                r = results[t]["reinforced"]
                writer.writerow([t, "Pure Epoxy Control", f"{c['peak_force_nN']:.3f}", f"{c['icss_MPa']:.2f}", f"{c['work_eV']:.2f}", f"{c['stiffness_nN_A']:.3f}"])
                writer.writerow([t, "MgO Reinforced", f"{r['peak_force_nN']:.3f}", f"{r['icss_MPa']:.2f}", f"{r['work_eV']:.2f}", f"{r['stiffness_nN_A']:.3f}"])
        print("Saved complete comparison summary to: comparison_summary.csv")

if __name__ == "__main__":
    main()
