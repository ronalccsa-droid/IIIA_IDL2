"""Recalcula el informe de monitoreo (PSI y CUSUM) a partir de las predicciones guardadas."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mcp.config import *
from mcp import features, drift as D

R = json.load(open(RESULTADOS / "resultados.json"))
df = pd.read_csv(RESULTADOS / "dataset.csv", parse_dates=["date"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
AZ, TE, GR = "#143278", "#A84B2F", "#7A7974"
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
R["drift"] = {}
for fam in FAMILIAS:
    Xf = features.construir(df, fam)
    cols = ["r_brent4", "r_fx4", "r_ca4"] if fam == "costo_mac" else ["r_brent4", "dist_plan", "util_plan"]
    ps = D.psi_movil(Xf[cols], INICIO_TEST)
    rc = pd.read_csv(RESULTADOS / f"predicciones_{fam}_{R['campeon'][fam][0]}.csv", parse_dates=["date"]).set_index("t")
    mon = D.monitor_desempeno(rc[rc.index >= INICIO_TEST], errores_ref=(rc.y - rc.q50)[rc.index < INICIO_TEST])
    mon.to_csv(RESULTADOS / f"monitoreo_{fam}.csv")
    R["drift"][fam] = dict(psi_max={c: float(ps[c].max()) for c in ps}, semanas_psi_mayor_025={c: int((ps[c] > 0.25).sum()) for c in ps},
                           primera_psi_025={c: (str(df.date.iloc[ps.index[ps[c] > 0.25][0]].date()) if (ps[c] > 0.25).any() else None) for c in ps},
                           alarmas_cusum=int(mon.alarma_cusum.sum()), semanas_cob_baja=int(mon.alarma_cobertura.sum()),
                           fechas_alarma_cusum=[str(x.date()) for x in mon[mon.alarma_cusum].date], cob_movil_min=float(mon.cob_movil.min()),
                           n_semanas=len(ps))
    if fam == "costo_mac":
        for c, col in zip(ps.columns, [AZ, TE, GR]):
            axs[0].plot(df.date.iloc[ps.index], ps[c], label=c, color=col)
        axs[0].axhline(0.25, ls="--", color="black", lw=0.7)
        axs[0].set_title("PSI móvil (26 semanas) de las variaciones a 4 semanas, MAC; umbral = 0,25", fontsize=8.5)
        axs[0].legend(fontsize=7, frameon=False)
    axs[1].plot(mon.date, mon.cob_movil, label=NOMBRE_FAMILIA[fam], color=AZ if fam == "costo_mac" else TE)
    for d_ in mon[mon.alarma_cusum].date:
        axs[1].axvline(d_, color=AZ if fam == "costo_mac" else TE, lw=0.6, ls=":")
axs[1].axhline(70, ls="--", color="black", lw=0.7)
axs[1].set_title("Cobertura móvil de 12 semanas (%); umbral = 70 %; líneas punteadas: alarmas CUSUM", fontsize=8.5)
axs[1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(FIGURAS / "fig_monitoreo.pdf"); plt.savefig(FIGURAS / "fig_monitoreo.png", dpi=200)
json.dump(R, open(RESULTADOS / "resultados.json", "w"), indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
print(json.dumps(R["drift"], indent=1))
