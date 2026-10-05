"""Pruebas de robustez y de falsación del método (Monte Carlo sobre el generador de datos).

Escenarios (20 réplicas cada uno, semillas distintas a la del prototipo):
  base      : misma estructura de costos, otros choques aleatorios.
  ruido_x2  : choques con el doble de desviación (estrés).
  caminata  : costos que son caminatas aleatorias sin relación con las variables (falsación):
              un método sin fuga de información NO debe superar al último costo.
La prueba de permutación de las variables explicativas está en scripts/permutacion.py.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mcp.config import *
from mcp import datos, backtest as B, metricas as M

REPLICAS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MOD = ["ultimo", "qr", "xgb"]
params = json.load(open(RESULTADOS / "resultados.json"))["hiperparametros"]
ext = datos.cargar_externas()
filas = []
for esc in ["base", "ruido_x2", "caminata"]:
    for k in range(REPLICAS):
        sem = 1000 + k
        df = datos.generar_sintetico(ext, semilla=sem, escenario="caminata" if esc == "caminata" else "base",
                                     ruido=2.0 if esc == "ruido_x2" else 1.0)
        for fam in FAMILIAS:
            res, _ = B.walk_forward(df, fam, MOD, params=params[fam], inicio=INICIO_TEST)
            b = res["ultimo"]
            for m in MOD[1:]:
                r = res[m]
                est, p = M.dm_hln((r.y - r.q50).values, (b.y - b.q50).values)
                filas.append(dict(escenario=esc, semilla=sem, familia=fam, modelo=m,
                                  mejora=100 * (1 - M.mae(r.y, r.q50) / M.mae(b.y, b.q50)), p=p, gana_sig=(p < 0.05) and est < 0))
        print(esc, k, flush=True)
D = pd.DataFrame(filas); D.to_csv(RESULTADOS / "robustez.csv", index=False)

res = {}
for (esc, fam, m), g in D.groupby(["escenario", "familia", "modelo"]):
    res[f"{esc}|{fam}|{m}"] = dict(mejora_media=g.mejora.mean(), p5=g.mejora.quantile(.05), p95=g.mejora.quantile(.95),
                                   pct_gana_sig=100 * g.gana_sig.mean())
json.dump(res, open(RESULTADOS / "robustez.json", "w"), indent=1, default=float)

import runpy; runpy.run_path(str(Path(__file__).resolve().parent / "fig_robustez.py"))
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 2)))
