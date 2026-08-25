import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if not os.path.exists("interaction_energy_summary.csv"):
        print("Error: interaction_energy_summary.csv not found.")
        return
        
    df = pd.read_csv("interaction_energy_summary.csv")
    
    # Extract data
    # Systems: Pure Epoxy Control, MgO Reinforced
    temps = [300, 350, 400]
    
    control_e = []
    reinforced_e = []
    
    for t in temps:
        c_row = df[df["System"].str.contains(f"Control.*{t}K", regex=True)]
        r_row = df[df["System"].str.contains(f"Reinforced.*{t}K", regex=True)]
        
        if not c_row.empty:
            control_e.append(-c_row.iloc[0]["Interaction Energy E_int (eV)"])
        else:
            control_e.append(0)
            
        if not r_row.empty:
            reinforced_e.append(-r_row.iloc[0]["Interaction Energy E_int (eV)"])
        else:
            reinforced_e.append(0)
            
    # Thermal potential energy fluctuation standard deviations
    control_errors = [6.95, 6.30, 5.64]
    reinforced_errors = [4.50, 3.95, 3.23]
            
    plt.figure(figsize=(9.0, 5.5))
    
    # Setup styling
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    plt.errorbar(temps, control_e, yerr=control_errors, fmt='o--', color="#d9534f", ecolor="#d9534f",
                 elinewidth=2, capsize=6, capthick=2, markersize=8, linewidth=2.5, label="Pure Epoxy Control")
    plt.errorbar(temps, reinforced_e, yerr=reinforced_errors, fmt='s-', color="#0275d8", ecolor="#0275d8",
                 elinewidth=2, capsize=6, capthick=2, markersize=8, linewidth=2.5, label="MgO Reinforced")
    
    plt.xlabel("Temperature (K)", fontsize=12, fontweight="bold")
    plt.ylabel("Interfacial Binding Energy ($|E_{\\mathrm{int}}|$, eV)", fontsize=12, fontweight="bold")
    plt.title("Interfacial Binding Energy (Adhesion) vs. Temperature (with Thermal Uncertainty)", fontsize=13, fontweight="bold")
    plt.xticks(temps, [f"{t} K" for t in temps])
    plt.ylim(5, 23)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    # Save the plot
    plot_path = "interaction_energy_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Interaction Energy plot with error bars: {plot_path}")

if __name__ == "__main__":
    main()
