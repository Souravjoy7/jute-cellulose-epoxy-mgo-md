import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Setup styling
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    # Load RDF data
    r_300 = pd.read_csv("reinforced_300K_rdf.csv")
    r_350 = pd.read_csv("reinforced_350K_rdf.csv")
    r_400 = pd.read_csv("reinforced_400K_rdf.csv")
    
    plt.figure(figsize=(8.0, 5.5))
    
    # Plot curves with distinct, bold colors
    plt.plot(r_300["r_A"], r_300["cell_O_mgo_Mg"], label="300 K (Peak: 1.98 Å)", color="#0275d8", linewidth=2.5)
    plt.plot(r_350["r_A"], r_350["cell_O_mgo_Mg"], label="350 K (Peak: 2.18 Å)", color="#f0ad4e", linewidth=2.2, linestyle="-.")
    plt.plot(r_400["r_A"], r_400["cell_O_mgo_Mg"], label="400 K (Peak: 2.28 Å)", color="#d9534f", linewidth=2.2, linestyle=":")
    
    plt.xlabel("Radial Distance $r$ ($\AA$)", fontsize=13, fontweight="bold")
    plt.ylabel("Radial Distribution Function $g(r)$", fontsize=13, fontweight="bold")
    plt.title("Temperature-Dependent Cellulose O – MgO Mg Coordinate RDF", fontsize=14, fontweight="bold")
    plt.xlim(1.5, 3.5)
    plt.ylim(-0.02, 0.42)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Highlight coordinate coordination peak range
    plt.axvspan(1.8, 2.4, color="#2ecc71", alpha=0.08, label="Coordination range")
    plt.legend(fontsize=11, loc="upper right")
    
    # Add clear arrow annotations pointing to the peaks with matching colors, positioned in free space at the top
    # 300K peak (1.975, 0.255) - Blue arrow pointing down-right from y=0.35, x=1.725
    plt.annotate("1.98 Å (300 K)\nPeak (g=0.26)", xy=(1.975, 0.255), xytext=(1.725, 0.35),
                 arrowprops=dict(facecolor='#0275d8', edgecolor='#0275d8', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 color='#0275d8', fontsize=10, fontweight="bold", ha="center")
                 
    # 350K peak (2.175, 0.210) - Orange arrow pointing down-right from y=0.35, x=2.15
    plt.annotate("2.18 Å (350 K)\nShift (g=0.21)", xy=(2.175, 0.2103), xytext=(2.15, 0.35),
                 arrowprops=dict(facecolor='#f0ad4e', edgecolor='#f0ad4e', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 color='#f0ad4e', fontsize=10, fontweight="bold", ha="center")
                 
    # 400K peak (2.275, 0.192) - Red arrow pointing down-left from y=0.35, x=2.575
    plt.annotate("2.28 Å (400 K)\nShift (g=0.19)", xy=(2.275, 0.1922), xytext=(2.575, 0.35),
                 arrowprops=dict(facecolor='#d9534f', edgecolor='#d9534f', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 color='#d9534f', fontsize=10, fontweight="bold", ha="center")
                 
    plt.tight_layout()
    plot_path = "mgo_coordinate_rdf_temperatures.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated standalone coordinate RDF plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
