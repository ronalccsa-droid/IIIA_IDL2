"""Prueba de permutación de la información explicativa (fuga de información y valor de las variables).

Se barajan en el tiempo, en bloque, todas las variables externas, de insumo y de planificación
(Brent, tipo de cambio, cemento asfáltico, diésel y cronograma de obra); se conservan el costo y su
historia. Si la ventaja del modelo proviene de información legítima, debe desaparecer al romper la
relación entre esas variables y el costo; si persistiera, indicaría fuga de información.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from mcp.config import *
from mcp import datos, backtest as B, metricas as M

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["qr"]
COLS = ["brent", "fx", "precio_ca", "diesel", "dist_acarreo_plan", "utilizacion_plan", "lluvia_plan", "prod_plan"]
params = json.load(open(RESULTADOS / "resultados.json"))["hiperparametros"]
ext = datos.cargar_externas()
df0 = datos.generar_sintetico(ext)
mej = lambda r, m: 100 * (1 - M.mae(r[m].y, r[m].q50) / M.mae(r["ultimo"].y, r["ultimo"].q50))
orig = {}
for fam in FAMILIAS:
    r = B.walk_forward(df0, fam, ["ultimo"] + MODS, params=params[fam], inicio=INICIO_TEST)[0]
    orig[fam] = {m: mej(r, m) for m in MODS}
filas = []
for k in range(N):
    rng = np.random.default_rng(500 + k)
    d = df0.copy(); idx = rng.permutation(len(d))
    d[COLS] = d[COLS].values[idx]
    for fam in FAMILIAS:
        r = B.walk_forward(d, fam, ["ultimo"] + MODS, params=params[fam], inicio=INICIO_TEST)[0]
        for m in MODS:
            filas.append(dict(familia=fam, modelo=m, mejora=mej(r, m)))
    print("perm", k, flush=True)
P = pd.DataFrame(filas); P.to_csv(RESULTADOS / "permutacion.csv", index=False)
res = {}
for (fam, m), g in P.groupby(["familia", "modelo"]):
    res[f"{fam}|{m}"] = dict(original=orig[fam][m], media=g.mejora.mean(), p95=g.mejora.quantile(.95),
                             maximo=g.mejora.max(), p_valor=(1 + (g.mejora >= orig[fam][m]).sum()) / (1 + len(g)))
R = json.load(open(RESULTADOS / "robustez.json")) if (RESULTADOS / "robustez.json").exists() else {}
R["permutacion"] = res
json.dump(R, open(RESULTADOS / "robustez.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
