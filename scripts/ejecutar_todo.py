"""Ejecuta el experimento completo y guarda resultados, figuras y artefactos.

Uso:  python scripts/ejecutar_todo.py                 (datos sintéticos calibrados, prototipo)
      python scripts/ejecutar_todo.py --erp ruta.csv  (historial real exportado del ERP)
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from mcp.config import *
from mcp import datos, features, backtest as B, metricas as M, contingencia as C, economico as E, drift as D, explicabilidad as X
from mcp.modelos import NOMBRES, TIPO, PARAMS_XGB, PARAMS_LGBM, CATALOGO

ap = argparse.ArgumentParser(); ap.add_argument("--erp", default=None); ap.add_argument("--rapido", action="store_true")
args = ap.parse_args()
t0 = time.time()
df = datos.cargar_erp(args.erp) if args.erp else datos.generar_sintetico()
calidad = datos.validar_esquema(df)
MODELOS = ["ultimo", "pm4", "ets", "sarimax", "arx", "qr", "xgb", "lgbm", "mlp", "lstm"] + ([] if args.rapido else ["chronos"])
R = dict(fuente="ERP" if args.erp else "sintetico_calibrado", calidad=calidad)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "savefig.dpi": 200})
AZ, TE, GR = "#143278", "#A84B2F", "#7A7974"
def guardar(nombre):
    plt.tight_layout(); plt.savefig(FIGURAS / f"{nombre}.pdf"); plt.savefig(FIGURAS / f"{nombre}.png"); plt.close()

# ================================================================ 1. EDA
eda = {}
for c in ["costo_mac", "costo_tierras", "brent", "fx", "precio_ca"]:
    s = df[c]
    eda[c] = dict(media=s.mean(), de=s.std(), min=s.min(), max=s.max(), cv=100 * s.std() / s.mean(),
                  adf_p_nivel=adfuller(s, autolag="AIC")[1], adf_p_dif=adfuller(np.log(s).diff().dropna(), autolag="AIC")[1])
vr = lambda s, q=4: np.log(s).diff(q).dropna().var() / (q * np.log(s).diff().dropna().var())
eda.update(vr_mac=vr(df.costo_mac), vr_tierras=vr(df.costo_tierras), n=len(df), inicio=str(df.date.iloc[0].date()),
           fin=str(df.date.iloc[-1].date()), inicio_test=str(df.date.iloc[INICIO_TEST].date()))
cor = df[["costo_mac", "costo_tierras", "precio_ca", "brent", "fx", "diesel", "prod_mac", "dist_acarreo", "utilizacion"]].corr()
eda["corr"] = cor.round(3).to_dict()
R["eda"] = eda

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].plot(df.date, df.brent, color=AZ); ax[0].set_title("Brent (US$ por barril)", fontsize=9)
ax[1].plot(df.date, df.fx, color=TE); ax[1].set_title("Tipo de cambio (S/ por US$)", fontsize=9)
for a in ax: a.tick_params(axis="x", rotation=30)
guardar("fig_externas")

# ================================================================ 2. Hiperparámetros (solo validación)
params, tablas_tun = {}, []
for fam in FAMILIAS:
    p, tab = B.ajustar_hiperparametros(df, fam)
    params[fam] = p; tab["familia"] = fam; tablas_tun.append(tab)
    print("hiperparámetros", fam, p, flush=True)
pd.concat(tablas_tun).to_csv(RESULTADOS / "ajuste_hiperparametros.csv", index=False)
R["hiperparametros"] = params

# ================================================================ 3. Walk-forward y evaluación
tablas, CAL, campeon, tiempos = [], {}, {}, {}
for fam in FAMILIAS:
    res, tm = B.walk_forward(df, fam, MODELOS, params=params[fam])
    tab, cal = B.evaluar(res, fam)
    tablas.append(tab); CAL[fam] = cal; tiempos[fam] = tm
    campeon[fam] = B.elegir_campeon(tab, fam)
    for m, r in cal.items():
        r.to_csv(RESULTADOS / f"predicciones_{fam}_{m}.csv")
    print("campeón", fam, campeon[fam], flush=True)
T = pd.concat(tablas); T["nombre"] = T.modelo.map(NOMBRES); T["tipo"] = T.modelo.map(TIPO)
T.to_csv(RESULTADOS / "metricas_modelos.csv", index=False)
R["campeon"] = campeon; R["tiempos_s"] = tiempos

# Conjunto de modelos de confianza (pérdida absoluta, periodo de prueba)
R["mcs"] = {}
for fam in FAMILIAS:
    L = pd.DataFrame({m: np.abs(r.y - r.q50)[r.index >= INICIO_TEST] for m, r in CAL[fam].items()})
    R["mcs"][fam] = M.mcs(L)

# Efecto mínimo detectable (potencia 80 %) con 54 semanas
R["mde"] = {}
for fam in FAMILIAS:
    r = CAL[fam][campeon[fam][0]]; b = CAL[fam]["ultimo"]; te = r.index >= INICIO_TEST
    R["mde"][fam] = M.efecto_minimo_detectable((r.y - r.q50)[te].values, (b.y - b.q50)[te].values)

# ================================================================ 4. Contingencia y economía
cont = {}
for fam in FAMILIAS:
    r = CAL[fam][campeon[fam][0]]; te = r[r.index >= INICIO_TEST]
    cont[fam] = C.backtest(te, CAL[fam]["ultimo"].q50)
R["contingencia"] = cont
esc = E.escenarios(cont); tor = E.tornado(cont); mc, vans = E.montecarlo(cont)
R["economico"] = dict(supuestos=E.SUPUESTOS, inversion=E.INVERSION, opex=E.OPEX, escenarios=esc, tornado=tor, montecarlo=mc)

# ================================================================ 5. Ablación (campeón probabilístico)
R["ablacion"] = {}
for fam in FAMILIAS:
    m = campeon[fam][0]
    if m in ("ens", "chronos", "ets", "sarimax"):
        m = "xgb"
    Xf = features.construir(df, fam); todas = list(Xf.columns)
    base = B.walk_forward(df, fam, [m], params=params[fam], cols=todas)[0][m]
    bte = base[base.index >= INICIO_TEST]; mae0 = M.mae(bte.y, bte.q50)
    fila = {"modelo": m, "Todas las variables": mae0}
    for g, cols in features.GRUPOS[fam].items():
        rr = B.walk_forward(df, fam, [m], params=params[fam], cols=[c for c in todas if c not in cols])[0][m]
        rte = rr[rr.index >= INICIO_TEST]
        fila[g] = M.mae(rte.y, rte.q50)
    R["ablacion"][fam] = fila
    print("ablación", fam, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)

# ================================================================ 6. Explicabilidad (SHAP con datos previos a la prueba)
R["shap"] = {}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
for a, fam in zip(axs, FAMILIAS):
    Xf, y, z, ref, ok = B.preparar(df, fam)
    idx = [i for i in range(INICIO_TEST - H + 1) if ok.iloc[i]]
    _, sv, _ = X.shap_xgb(Xf.iloc[idx], z.iloc[idx], params[fam]["xgb"])
    imp = sv.abs().mean().sort_values()
    R["shap"][fam] = (100 * imp / imp.sum()).sort_values(ascending=False).round(2).to_dict()
    a.barh(imp.index, 100 * imp / imp.sum(), color=AZ); a.set_title(f"{NOMBRE_FAMILIA[fam]}\n|SHAP| medio (% del total)", fontsize=8.5)
guardar("fig_shap")
# dependencia SHAP de la variable principal
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
for a, fam in zip(axs, FAMILIAS):
    Xf, y, z, ref, ok = B.preparar(df, fam)
    idx = [i for i in range(INICIO_TEST - H + 1) if ok.iloc[i]]
    _, sv, _ = X.shap_xgb(Xf.iloc[idx], z.iloc[idx], params[fam]["xgb"])
    top = sv.abs().mean().idxmax()
    a.scatter(Xf.iloc[idx][top], sv[top], s=10, color=TE); a.axhline(0, color=GR, lw=0.6)
    a.set_xlabel(top); a.set_ylabel("Valor SHAP (log)"); a.set_title(NOMBRE_FAMILIA[fam], fontsize=8.5)
guardar("fig_shap_dependencia")

# ================================================================ 7. Curva de aprendizaje (tamaño de la ventana)
R["curva_aprendizaje"] = {}
for fam in FAMILIAS:
    m = campeon[fam][0] if campeon[fam][0] in ("qr", "xgb", "lgbm", "arx") else "xgb"
    fila = {}
    for v in [20, 30, 40, 50, None]:
        rr = B.walk_forward(df, fam, [m], params=params[fam], ventana=v, inicio=INICIO_TEST)[0][m]
        fila[str(v or "expansiva")] = M.mae(rr.y, rr.q50)
    R["curva_aprendizaje"][fam] = dict(modelo=m, **fila)

# ================================================================ 8. Deriva y monitoreo: scripts/monitoreo.py (al final)

# ================================================================ 9. Figuras de resultados
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8))
for a, fam in zip(axs, FAMILIAS):
    r = CAL[fam][campeon[fam][0]]; te = r[r.index >= INICIO_TEST]; b = CAL[fam]["ultimo"].loc[te.index]
    a.fill_between(te.date, te.lo, te.hi, color=AZ, alpha=0.15, label="Intervalo P10–P90 calibrado")
    a.plot(te.date, te.y, color="black", lw=1.2, label="Costo real")
    a.plot(te.date, te.q50, color=AZ, lw=1.1, label=f"Motor: {NOMBRES[campeon[fam][0]]} (P50)")
    a.plot(te.date, b.q50, color=TE, lw=1, ls="--", label="Último costo (línea de base)")
    a.set_title(NOMBRE_FAMILIA[fam], fontsize=9); a.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")
guardar("fig_predicciones")

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
for a, fam in zip(axs, FAMILIAS):
    t_ = T[(T.familia == fam) & (T.periodo == "prueba")].sort_values("MAPE", ascending=False)
    col = [TE if m == "ultimo" else (AZ if m == campeon[fam][0] else GR) for m in t_.modelo]
    a.barh(t_.nombre, t_.MAPE, color=col)
    for i, v in enumerate(t_.MAPE): a.text(v, i, f" {v:.2f}", va="center", fontsize=6.5)
    a.set_title(f"{NOMBRE_FAMILIA[fam]}: MAPE (%)", fontsize=8.5); a.tick_params(axis="y", labelsize=6.5)
guardar("fig_errores")

items = sorted(tor.items(), key=lambda kv: abs(kv[1][1] - kv[1][0])); vb = esc["Base"]["VAN"]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
for i, (lab, (lo_, hi_)) in enumerate(items):
    ax.barh(i, lo_ - vb, left=vb, color=TE, height=0.55); ax.barh(i, hi_ - vb, left=vb, color=AZ, height=0.55)
ax.axvline(vb, color="black", lw=0.8); ax.axvline(0, color=GR, lw=0.8, ls="--")
ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items]); ax.set_xlabel("VAN (S/)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f} mil"))
ax.set_title("Sensibilidad del VAN: ±50 % (rojo: desfavorable; azul: favorable)", fontsize=9)
guardar("fig_tornado")

fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.hist(vans / 1000, bins=60, color=AZ, alpha=0.8); ax.axvline(0, color=TE, lw=1.2)
ax.set_xlabel("VAN (miles de S/)"); ax.set_ylabel("Frecuencia")
ax.set_title("Monte Carlo (10 000 escenarios): P(VAN > 0) = " + f"{mc['prob_van_pos']:.1f}".replace(".", ",") + " %", fontsize=9)
guardar("fig_montecarlo_van")

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
for a, fam in zip(axs, FAMILIAS):
    ab = R["ablacion"][fam]; base_ = ab["Todas las variables"]
    ks = [k for k in ab if k not in ("modelo", "Todas las variables")]
    vals = [100 * (ab[k] / base_ - 1) for k in ks]
    a.barh(ks, vals, color=[TE if v > 0 else AZ for v in vals]); a.axvline(0, color="black", lw=0.6)
    a.set_title(f"{NOMBRE_FAMILIA[fam]}\nAumento del MAE al quitar el grupo (%)", fontsize=8.5); a.tick_params(axis="y", labelsize=7)
guardar("fig_ablacion")

# ================================================================ 10. Artefactos para el tablero
import xgboost as xgb, lightgbm as lgb
art = dict(fecha=str(df.date.iloc[-1].date()), params=params, campeon=campeon, factores=C.FACTOR_CONTRATO, familias={})
for fam in FAMILIAS:
    Xf, y, z, ref, ok = B.preparar(df, fam)
    idx = [i for i in range(len(df)) if ok.iloc[i]]
    mx = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(CUANTILES), random_state=0,
                          **dict(PARAMS_XGB, **params[fam]["xgb"])).fit(Xf.iloc[idx], z.iloc[idx])
    r = CAL[fam][campeon[fam][0]]
    art["familias"][fam] = dict(modelo_xgb=mx, columnas=list(Xf.columns), ultimo_qhat=float(r.qhat.iloc[-1]),
                                historia=df[["date", fam]].rename(columns={fam: "costo"}), X=Xf,
                                ref=ref, predicciones=r, metricas=T[T.familia == fam])
joblib.dump(art, ARTEFACTOS / "motor.joblib")
df.to_csv(RESULTADOS / "dataset.csv", index=False)
json.dump(R, open(RESULTADOS / "resultados.json", "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
import runpy; runpy.run_path(str(Path(__file__).resolve().parent / "monitoreo.py"))
print(f"\nListo en {time.time() - t0:.0f} s. Campeones: {campeon}")
