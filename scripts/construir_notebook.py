"""Genera notebooks/motor_costos_predictivos_colab.ipynb (autocontenido: incluye el paquete y los datos)."""
from pathlib import Path
import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Motor de costos predictivos para la cotización de obras viales
**Maestría en Ciencia de Datos e Inteligencia Artificial · Caso INDUMAQ S.R.L. (Arequipa, Perú)**

Este cuaderno reproduce de principio a fin el experimento del informe: datos, análisis exploratorio, ajuste de
hiperparámetros, comparación de 12 modelos con validación walk-forward, intervalos conformales, pruebas de
significancia, ablación, explicabilidad (SHAP), monitoreo de deriva, pruebas de robustez y análisis económico.

**Cómo usarlo en Google Colab:** `Entorno de ejecución → Ejecutar todo`. Tarda entre 8 y 12 minutos en CPU.
Para usar los datos reales de la empresa, suba un CSV con el esquema de `data/plantilla_erp.csv` en la sección 1.

| Paso | Contenido |
|---|---|
| 0 | Instalación y código del paquete `mcp` |
| 1 | Datos: variables externas reales (Brent, USD/PEN) y costos internos |
| 2 | Análisis exploratorio |
| 3 | Ajuste de hiperparámetros (solo periodo de validación) |
| 4 | Validación walk-forward de 12 modelos |
| 5 | Métricas, pruebas DM-HLN con corrección de Holm, bootstrap y MCS |
| 6 | Contingencia y comparación con la línea de base |
| 7 | Ablación y explicabilidad SHAP |
| 8 | Monitoreo de deriva |
| 9 | Robustez y falsación |
| 10 | Análisis económico y Monte Carlo |
""")
md("## 0. Instalación y código fuente")
code("""
import sys, subprocess, importlib.util
paquetes = ["lightgbm", "arch", "chronos-forecasting", "xgboost", "statsmodels"]
faltan = [p for p in paquetes if importlib.util.find_spec(p.replace("-forecasting", "").replace("-", "_")) is None]
if faltan:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan], check=True)
import os; os.makedirs("src/mcp", exist_ok=True); os.makedirs("data", exist_ok=True)
""")
md("El paquete `mcp` se escribe en disco con las celdas siguientes, para que el código quede visible y versionado en el propio cuaderno.")
for f in ["__init__.py", "config.py", "datos.py", "features.py", "modelos.py", "conformal.py", "metricas.py", "backtest.py",
          "contingencia.py", "drift.py", "explicabilidad.py", "economico.py"]:
    txt = (RAIZ / "src/mcp" / f).read_text(encoding="utf-8")
    code(f"%%writefile src/mcp/{f}\n" + (txt if txt.strip() else "# Paquete mcp\n"))
code("%%writefile data/externas_semanal.csv\n" + (RAIZ / "data/externas_semanal.csv").read_text())
code("%%writefile data/plantilla_erp.csv\n" + (RAIZ / "data/plantilla_erp.csv").read_text())
code("""
import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mcp.config import *
from mcp import datos, features, backtest as B, metricas as M, contingencia as C, economico as E, drift as D, explicabilidad as X
from mcp.modelos import NOMBRES, TIPO
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
""")
md("""
## 1. Datos
Variables externas **reales**: Brent semanal (US$ por barril) y tipo de cambio interbancario USD/PEN.
Costos internos: si `RUTA_ERP` apunta al CSV exportado del ERP se usan los datos reales; si no, se usan costos
**sintéticos calibrados** con la estructura de costos de la empresa (cemento asfáltico ≈ 53 % del costo de la MAC).
""")
code("""
RUTA_ERP = None   # p. ej. "historial_erp.csv"; en Colab: from google.colab import files; files.upload()
df = datos.cargar_erp(RUTA_ERP) if RUTA_ERP else datos.generar_sintetico()
print("Fuente:", "ERP" if RUTA_ERP else "sintética calibrada", "| semanas:", len(df), "|", df.date.min().date(), "a", df.date.max().date())
print("Control de calidad:", datos.validar_esquema(df))
df.head()
""")
md("## 2. Análisis exploratorio")
code("""
from statsmodels.tsa.stattools import adfuller
filas = []
for c in ["costo_mac", "costo_tierras", "precio_ca", "brent", "fx"]:
    s = df[c]
    filas.append(dict(serie=c, media=s.mean(), de=s.std(), cv=100*s.std()/s.mean(),
                      adf_p_nivel=adfuller(s, autolag="AIC")[1], adf_p_dif=adfuller(np.log(s).diff().dropna(), autolag="AIC")[1]))
display(pd.DataFrame(filas).round(3))
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
ax[0].plot(df.date, df.brent); ax[0].set_title("Brent (US$/bbl)")
ax[1].plot(df.date, df.costo_mac, color="C1"); ax[1].set_title("Costo MAC (S/ por t)")
ax[2].plot(df.date, df.costo_tierras, color="C2"); ax[2].set_title("Costo tierras (S/ por m³)")
plt.tight_layout(); plt.show()
df[["costo_mac", "costo_tierras", "precio_ca", "brent", "fx", "diesel", "prod_mac", "dist_acarreo", "utilizacion"]].corr().round(2)
""")
md("""
## 3. Ajuste de hiperparámetros
Búsqueda en rejilla con origen móvil **solo en el periodo de validación** (semanas 61 a 84). El periodo de prueba
no interviene en ninguna decisión. Criterio: pérdida cuantílica media de P10, P50 y P90.
""")
code("""
params = {}
for fam in FAMILIAS:
    params[fam], tab = B.ajustar_hiperparametros(df, fam)
    print(fam, params[fam])
""")
md("""
## 4. Validación walk-forward de 12 modelos
Cada semana *t* se reentrena con los objetivos ya observados (semanas ≤ *t* − 4) y se predice el costo medio de
las 4 semanas siguientes. Los intervalos de todos los modelos probabilísticos se calibran con CQR + ACI.
""")
code("""
MODELOS = ["ultimo", "pm4", "ets", "sarimax", "arx", "qr", "xgb", "lgbm", "mlp", "lstm", "chronos"]
T, CAL, campeon = [], {}, {}
for fam in FAMILIAS:
    res, tiempos = B.walk_forward(df, fam, MODELOS, params=params[fam])
    tab, CAL[fam] = B.evaluar(res, fam)
    T.append(tab); campeon[fam] = B.elegir_campeon(tab, fam)
    print(fam, "campeón probabilístico / puntual (elegidos en validación):", campeon[fam])
T = pd.concat(T); T["nombre"] = T.modelo.map(NOMBRES)
""")
md("## 5. Resultados en el periodo de prueba (54 semanas)")
code("""
cols = ["nombre", "MAE", "MAPE", "MASE", "mejora_MAE", "IC95_inf", "IC95_sup", "p_DM", "p_Holm", "cob_cal", "kupiec_p", "winkler"]
for fam in FAMILIAS:
    print(NOMBRE_FAMILIA[fam]); display(T[(T.familia == fam) & (T.periodo == "prueba")][cols].round(3))
""")
code("""
# Conjunto de modelos de confianza (Hansen, Lunde y Nason, 2011) al 90 %
for fam in FAMILIAS:
    L = pd.DataFrame({m: np.abs(r.y - r.q50)[r.index >= INICIO_TEST] for m, r in CAL[fam].items()})
    p = pd.Series(M.mcs(L)).sort_values(ascending=False)
    print(NOMBRE_FAMILIA[fam], "→ en el MCS:", list(p[p >= 0.10].index), "| excluidos:", list(p[p < 0.10].index))
    r = CAL[fam][campeon[fam][0]]; b = CAL[fam]["ultimo"]; te = r.index >= INICIO_TEST
    print("   efecto mínimo detectable (80 %% de potencia): %.1f %%" % M.efecto_minimo_detectable((r.y-r.q50)[te].values, (b.y-b.q50)[te].values))
""")
code("""
fig, axs = plt.subplots(2, 1, figsize=(12, 6))
for a, fam in zip(axs, FAMILIAS):
    r = CAL[fam][campeon[fam][0]]; te = r[r.index >= INICIO_TEST]; b = CAL[fam]["ultimo"].loc[te.index]
    a.fill_between(te.date, te.lo, te.hi, alpha=0.2, label="P10–P90 calibrado")
    a.plot(te.date, te.y, "k", label="Real"); a.plot(te.date, te.q50, label=NOMBRES[campeon[fam][0]])
    a.plot(te.date, b.q50, "--", label="Último costo"); a.set_title(NOMBRE_FAMILIA[fam]); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
md("## 6. Contingencia: práctica actual (último costo + 5 %) frente al motor")
code("""
cont = {}
for fam in FAMILIAS:
    r = CAL[fam][campeon[fam][0]]
    cont[fam] = C.backtest(r[r.index >= INICIO_TEST], CAL[fam]["ultimo"].q50)
pd.DataFrame(cont).round(3)
""")
md("## 7. Ablación y explicabilidad")
code("""
for fam in FAMILIAS:
    m = campeon[fam][0] if campeon[fam][0] in ("qr", "xgb", "lgbm", "arx") else "xgb"
    Xf = features.construir(df, fam); todas = list(Xf.columns)
    base = B.walk_forward(df, fam, [m], params=params[fam], inicio=INICIO_TEST)[0][m]; mae0 = M.mae(base.y, base.q50)
    out = {}
    for g, cols_g in features.GRUPOS[fam].items():
        rr = B.walk_forward(df, fam, [m], params=params[fam], cols=[c for c in todas if c not in cols_g], inicio=INICIO_TEST)[0][m]
        out[g] = 100 * (M.mae(rr.y, rr.q50) / mae0 - 1)
    print(NOMBRE_FAMILIA[fam], f"({NOMBRES[m]}): aumento del MAE al quitar cada grupo (%)"); display(pd.Series(out).round(1))
""")
code("""
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for a, fam in zip(axs, FAMILIAS):
    Xf, y, z, ref, ok = B.preparar(df, fam)
    idx = [i for i in range(INICIO_TEST - H + 1) if ok.iloc[i]]
    _, sv, _ = X.shap_xgb(Xf.iloc[idx], z.iloc[idx], params[fam]["xgb"])
    imp = sv.abs().mean().sort_values(); a.barh(imp.index, 100 * imp / imp.sum()); a.set_title(f"{NOMBRE_FAMILIA[fam]}: |SHAP| medio (%)")
plt.tight_layout(); plt.show()
""")
md("## 8. Monitoreo de deriva (PSI y CUSUM)")
code("""
for fam in FAMILIAS:
    Xf = features.construir(df, fam)
    cols_psi = ["r_brent4", "r_fx4", "r_ca4"] if fam == "costo_mac" else ["r_brent4", "dist_plan", "util_plan"]
    ps = D.psi_movil(Xf[cols_psi], INICIO_TEST)
    rc = CAL[fam][campeon[fam][0]]
    mon = D.monitor_desempeno(rc[rc.index >= INICIO_TEST], errores_ref=(rc.y - rc.q50)[rc.index < INICIO_TEST])
    print(NOMBRE_FAMILIA[fam], "| PSI máximo:", ps.max().round(2).to_dict(), "| alarmas CUSUM:", int(mon.alarma_cusum.sum()),
          "| cobertura móvil mínima: %.0f %%" % mon.cob_movil.min())
""")
md("""
## 9. Robustez y falsación
Se repite el experimento con otras semillas del generador (escenario base), con el doble de ruido y con costos que
son caminatas aleatorias (falsación: el modelo **no** debe ganar). `REPLICAS = 5` para Colab; el informe usa 20.
""")
code("""
REPLICAS = 5
filas = []
for esc in ["base", "ruido_x2", "caminata"]:
    for k in range(REPLICAS):
        d = datos.generar_sintetico(semilla=1000 + k, escenario="caminata" if esc == "caminata" else "base", ruido=2.0 if esc == "ruido_x2" else 1.0)
        for fam in FAMILIAS:
            res, _ = B.walk_forward(d, fam, ["ultimo", "qr", "xgb"], params=params[fam], inicio=INICIO_TEST)
            for m in ["qr", "xgb"]:
                filas.append(dict(escenario=esc, familia=fam, modelo=m, mejora=100 * (1 - M.mae(res[m].y, res[m].q50) / M.mae(res["ultimo"].y, res["ultimo"].q50))))
pd.DataFrame(filas).groupby(["escenario", "familia", "modelo"]).mejora.describe()[["mean", "min", "max"]].round(1)
""")
md("""
**Permutación de la información explicativa.** Se barajan en el tiempo, en bloque, las variables externas, de insumo y
de planificación; si la ventaja proviene de información legítima, debe desaparecer (el informe usa 20 permutaciones).
""")
code("""
COLS = ["brent", "fx", "precio_ca", "diesel", "dist_acarreo_plan", "utilizacion_plan", "lluvia_plan", "prod_plan"]
filas = []
for k in range(5):
    d = df.copy(); idx = np.random.default_rng(500 + k).permutation(len(d)); d[COLS] = d[COLS].values[idx]
    for fam in FAMILIAS:
        res, _ = B.walk_forward(d, fam, ["ultimo", "qr"], params=params[fam], inicio=INICIO_TEST)
        filas.append(dict(familia=fam, mejora=100 * (1 - M.mae(res["qr"].y, res["qr"].q50) / M.mae(res["ultimo"].y, res["ultimo"].q50))))
pd.DataFrame(filas).groupby("familia").mejora.describe()[["mean", "min", "max"]].round(1)
""")
md("## 10. Análisis económico (VAN, TIR, B/C y Monte Carlo)")
code("""
esc = E.escenarios(cont)
display(pd.DataFrame({k: dict(VAN=v["VAN"], TIR=100 * v["TIR"], BC=v["BC"], recuperacion=v["recuperacion"]) for k, v in esc.items()}).round(2))
mc, vans = E.montecarlo(cont)
print("Monte Carlo:", {k: round(v, 1) for k, v in mc.items()})
plt.hist(vans / 1000, bins=60); plt.axvline(0, color="r"); plt.xlabel("VAN (miles de S/)"); plt.show()
""")
md("""
## Conclusión del cuaderno
Los resultados numéricos de este cuaderno coinciden con los del informe cuando se usan los datos sintéticos con la
semilla 2026. Con el historial real del ERP, basta con definir `RUTA_ERP` y ejecutar todo de nuevo: el diseño de
validación, la selección del campeón y las pruebas no cambian.
""")
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "colab": {"provenance": [], "toc_visible": True}}
(RAIZ / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, RAIZ / "notebooks/motor_costos_predictivos_colab.ipynb")
print("ok")
