"""Figura de robustez a partir de resultados/robustez.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mcp.config import *
D = pd.read_csv(RESULTADOS / "robustez.csv")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
nom = {"qr": "QR", "xgb": "XGB"}; esc_n = {"base": "Base", "ruido_x2": "Ruido ×2", "caminata": "Falsación"}
tit = {"costo_mac": "MAC", "costo_tierras": "Movimiento de tierras"}
for a, fam in zip(axs, FAMILIAS):
    data, labs = [], []
    for e in ["base", "ruido_x2", "caminata"]:
        for m in ["qr", "xgb"]:
            data.append(D[(D.escenario == e) & (D.familia == fam) & (D.modelo == m)].mejora.values); labs.append(f"{esc_n[e]}\n{nom[m]}")
    a.boxplot(data, widths=0.6, medianprops=dict(color="#A84B2F"))
    a.set_xticks(range(1, len(labs) + 1), labs, fontsize=7)
    a.axhline(0, color="black", lw=0.6); a.set_title(tit[fam], fontsize=9); a.set_ylabel("Mejora del MAE frente al último costo (%)", fontsize=7.5)
plt.tight_layout(); plt.savefig(FIGURAS / "fig_robustez.pdf"); plt.savefig(FIGURAS / "fig_robustez.png", dpi=200)
