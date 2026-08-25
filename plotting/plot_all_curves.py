import os
import csv
import matplotlib.pyplot as plt

def read_csv(csv_path):
    displacements = []
    forces_nN = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            displacements.append(float(row[1]))
            # 1 eV/A = 1.602176634 nN
            forces_nN.append(float(row[2]) * 1.602176634)
    return displacements, forces_nN

def main():
    # Define files and styles
    files = {
        "Control (300 K)": ("control/pullout_data.csv", "#d9534f", "--"),
        "MgO Reinforced (300 K)": ("reinforced/pullout_data.csv", "#0275d8", "-"),
        "Control (350 K)": ("control_350K/pullout_data.csv", "#f0ad4e", "--"),
        "MgO Reinforced (350 K)": ("reinforced_350K/pullout_data.csv", "#5bc0de", "-"),
        "Control (400 K)": ("control_400K/pullout_data.csv", "#9b59b6", "--"),
        "MgO Reinforced (400 K)": ("reinforced_400K/pullout_data.csv", "#5cb85c", "-")
    }
    
    plt.figure(figsize=(11.5, 6.5))
    
    for label, (path, color, style) in files.items():
        if os.path.exists(path):
            disp, force = read_csv(path)
            # Smooth curves slightly for cleaner visual output
            plt.plot(disp, force, label=label, color=color, linestyle=style, linewidth=2.2)
        else:
            print(f"Skipping {label} ({path} not found)")
            
    plt.xlabel("Displacement ($\AA$)", fontsize=13, fontweight="bold")
    plt.ylabel("Pull-out Force (nN)", fontsize=13, fontweight="bold")
    plt.title("Steered Molecular Dynamics Pull-out Force vs. Displacement (All Systems)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    plot_path = "pullout_curves_comparison_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated comparative pull-out plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
