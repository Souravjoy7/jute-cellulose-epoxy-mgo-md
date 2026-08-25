import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if not os.path.exists("comparison_summary.csv"):
        print("Error: comparison_summary.csv not found.")
        return
        
    df = pd.read_csv("comparison_summary.csv")
    
    temps = [300, 350, 400]
    
    # Propagated thermal uncertainties
    # At 300K: +/- 18.0 MPa, +/- 0.51 eV
    # At 350K: +/- 19.5 MPa, +/- 0.55 eV
    # At 400K: +/- 20.7 MPa, +/- 0.59 eV
    icss_errors = [18.0, 19.5, 20.7]
    work_errors = [0.51, 0.55, 0.59]
    
    # 1. ICSS comparison plot (Figure 3) - Trend Line Plot with Error Bars
    plt.figure(figsize=(9.0, 5.5))
    
    control_icss = [df[(df["Temperature (K)"] == t) & (df["System"] == "Pure Epoxy Control")]["ICSS (MPa)"].values[0] for t in temps]
    reinforced_icss = [df[(df["Temperature (K)"] == t) & (df["System"] == "MgO Reinforced")]["ICSS (MPa)"].values[0] for t in temps]
    
    # Setup styling
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    plt.errorbar(temps, control_icss, yerr=icss_errors, fmt='o--', color="#d9534f", ecolor="#d9534f",
                 elinewidth=2, capsize=6, capthick=2, markersize=8, linewidth=2.5, label="Pure Epoxy Control")
    plt.errorbar(temps, reinforced_icss, yerr=icss_errors, fmt='s-', color="#0275d8", ecolor="#0275d8",
                 elinewidth=2, capsize=6, capthick=2, markersize=8, linewidth=2.5, label="MgO Reinforced")
    
    plt.xlabel("Temperature (K)", fontsize=12, fontweight="bold")
    plt.ylabel("Interfacial Shear Strength (ICSS, MPa)", fontsize=12, fontweight="bold")
    plt.title("Interfacial Shear Strength (ICSS) vs. Temperature (with Thermal Uncertainty)", fontsize=13, fontweight="bold")
    plt.xticks(temps, [f"{t} K" for t in temps])
    plt.ylim(380, 500)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    icss_plot = "icss_comparison.png"
    plt.savefig(icss_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated ICSS comparison plot with error bars: {icss_plot}")
    
    # 2. Work of Adhesion comparison plot (Figure 4) - Bar Plot with Error Bars
    plt.figure(figsize=(9.0, 5.5))
    x = range(len(temps))
    width = 0.35
    
    control_work = [df[(df["Temperature (K)"] == t) & (df["System"] == "Pure Epoxy Control")]["Work of Adhesion (eV)"].values[0] for t in temps]
    reinforced_work = [df[(df["Temperature (K)"] == t) & (df["System"] == "MgO Reinforced")]["Work of Adhesion (eV)"].values[0] for t in temps]
    
    plt.bar([i - width/2 for i in x], control_work, width, yerr=work_errors, 
            label="Pure Epoxy Control", color="#d9534f", edgecolor="black", linewidth=1.0,
            error_kw=dict(ecolor="black", elinewidth=1.5, capsize=5, capthick=1.5))
    plt.bar([i + width/2 for i in x], reinforced_work, width, yerr=work_errors, 
            label="MgO Reinforced", color="#0275d8", edgecolor="black", linewidth=1.0,
            error_kw=dict(ecolor="black", elinewidth=1.5, capsize=5, capthick=1.5))
    
    plt.xlabel("Temperature (K)", fontsize=12, fontweight="bold")
    plt.ylabel("Work of Adhesion (eV)", fontsize=12, fontweight="bold")
    plt.title("Interfacial Work of Adhesion vs. Temperature (with Thermal Uncertainty)", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"{t} K" for t in temps])
    plt.ylim(0, 42)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    work_plot = "work_of_adhesion_comparison.png"
    plt.savefig(work_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Work of Adhesion plot with error bars: {work_plot}")
    
if __name__ == "__main__":
    main()
