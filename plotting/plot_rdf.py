import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Setup styling
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    # Load RDF data
    c_300 = pd.read_csv("control_300K_rdf.csv")
    r_300 = pd.read_csv("reinforced_300K_rdf.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))
    
    # Panel 1: Interfacial Hydrogen Bonding RDF (300 K)
    # 1. Cellulose H - MgO O (Reinforced)
    if "cell_H_mgo_O" in r_300.columns:
        ax1.plot(r_300["r_A"], r_300["cell_H_mgo_O"], label="Cellulose H – MgO O (Reinforced)", color="#0275d8", linewidth=2.5)
    # 2. Cellulose H - Epoxy O (Reinforced)
    if "cell_H_epoxy_O" in r_300.columns:
        ax1.plot(r_300["r_A"], r_300["cell_H_epoxy_O"], label="Cellulose H – Epoxy O (Reinforced)", color="#5cb85c", linewidth=2.0)
    # 3. Cellulose H - Epoxy O (Control)
    if "cell_H_epoxy_O" in c_300.columns:
        ax1.plot(c_300["r_A"], c_300["cell_H_epoxy_O"], label="Cellulose H – Epoxy O (Control)", color="#d9534f", linewidth=2.0, linestyle="--")
        
    ax1.set_xlabel("Distance $r$ ($\AA$)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Radial Distribution Function $g(r)$", fontsize=13, fontweight="bold")
    ax1.set_title("(a) Interfacial Hydrogen Bonding RDF (300 K)", fontsize=14, fontweight="bold")
    ax1.set_xlim(1.0, 5.0)
    ax1.set_ylim(-0.05, 1.4)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Highlight the primary H-bonding peak region (1.5 - 3.0 Å)
    ax1.axvspan(1.5, 3.0, color="#f1c40f", alpha=0.1, label="H-bonding region")
    ax1.legend(fontsize=10.5, loc="upper left")
    
    # Annotate Hydrogen Bonding Peaks
    ax1.annotate("2.58 Å\nControl Peak\n(g=0.96)", xy=(2.575, 0.9644), xytext=(3.3, 1.15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 fontsize=9.5, fontweight="bold", ha="center")
                 
    ax1.annotate("2.78 Å\nReinforced Peak\n(g=0.55)", xy=(2.775, 0.5536), xytext=(3.7, 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 fontsize=9.5, fontweight="bold", ha="center")

    # Panel 2: Cellulose–MgO Coordinate Ionic Bonding RDF (Mg-O) at 300 K
    if "cell_O_mgo_Mg" in r_300.columns:
        ax2.plot(r_300["r_A"], r_300["cell_O_mgo_Mg"], label="MgO Coordinate Bond (300 K)", color="#8e44ad", linewidth=2.5)
        
    ax2.set_xlabel("Distance $r$ ($\AA$)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Radial Distribution Function $g(r)$", fontsize=13, fontweight="bold")
    ax2.set_title("(b) Cellulose O – MgO Mg Coordinate Bonding RDF (300 K)", fontsize=14, fontweight="bold")
    ax2.set_xlim(1.5, 5.0)
    ax2.set_ylim(-0.02, 0.45)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Highlight the Mg-O coordinate bond peak region (1.8 - 2.4 Å)
    ax2.axvspan(1.8, 2.4, color="#2ecc71", alpha=0.08, label="Mg-O coordination shell")
    ax2.legend(fontsize=10.5, loc="upper right")
    
    # Annotate Coordinate Bond Peak at 300 K
    ax2.annotate("1.98 Å (300 K)\nPeak (g=0.26)", xy=(1.975, 0.255), xytext=(2.7, 0.32),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5, headlength=5),
                 fontsize=9.5, fontweight="bold", ha="center")
    
    plt.tight_layout()
    plot_path = "rdf_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated publication-quality RDF plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
