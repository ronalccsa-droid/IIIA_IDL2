"""Tablero del motor de costos predictivos (Streamlit).

Ejecutar:  streamlit run app/tablero.py
Requiere haber corrido antes:  python scripts/ejecutar_todo.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from mcp.config import RESULTADOS, ARTEFACTOS, H, INICIO_TEST, NOMBRE_FAMILIA
from mcp import features, backtest as B, contingencia as C
from mcp.modelos import Contexto, CATALOGO, NOMBRES, MIEMBROS_ENSAMBLE
from mcp.explicabilidad import shap_xgb

st.set_page_config(page_title="Motor de costos predictivos", layout="wide")


@st.cache_resource
def cargar():
    art = joblib.load(ARTEFACTOS / "motor.joblib")
    df = pd.read_csv(RESULTADOS / "dataset.csv", parse_dates=["date"])
    tab = pd.read_csv(RESULTADOS / "metricas_modelos.csv")
    return art, df, tab


art, df, tab = cargar()
st.title("Motor de costos predictivos")
st.caption(f"INDUMAQ S.R.L. · Datos hasta {art['fecha']} · Horizonte: {H} semanas · Intervalo P10–P90 calibrado (CQR + ACI)")

with st.sidebar:
    st.header("Cotización")
    qp = st.query_params
    fams = list(NOMBRE_FAMILIA)
    fam = st.selectbox("Familia de costo", fams, index=fams.index(qp.get("familia", fams[0])), format_func=NOMBRE_FAMILIA.get)
    tipo = st.selectbox("Tipo de contrato", list(C.FACTOR_CONTRATO), index=int(qp.get("contrato", 0)))
    meses = st.slider("Plazo de ejecución (meses)", 1, 12, int(qp.get("meses", 1)))
    tope = st.slider("Tope de contingencia (%)", 5, 40, 25) / 100
    cantidad = st.number_input("Cantidad a cotizar (t o m³)", min_value=0.0, value=10_000.0, step=1_000.0)
    st.header("Escenario de mercado")
    sh_b = st.slider("Variación del Brent frente a hoy (%)", -30, 30, 0)
    sh_fx = st.slider("Variación del tipo de cambio (%)", -10, 10, 0)
    st.header("Planificación de obra")
    if fam == "costo_tierras":
        dist = st.slider("Distancia media de acarreo (km)", 1.0, 8.0, float(round(df.dist_acarreo.tail(8).mean(), 1)))
        util = st.slider("Utilización de flota", 0.45, 0.95, float(round(df.utilizacion.tail(8).mean(), 2)))
        lluvia = st.select_slider("Semanas en temporada de lluvias", options=[0, 1, 2, 3, 4], value=int(df.lluvia.tail(4).sum()))
    else:
        prod = st.slider("Producción planificada (t por semana)", 1400, 4000, int(df.prod_mac.tail(8).mean()))

# ---------------------------------------------------------------- predicción con el campeón
d = df.copy(); t = len(d) - 1
d.loc[t, "brent"] *= 1 + sh_b / 100; d.loc[t, "fx"] *= 1 + sh_fx / 100
if fam == "costo_tierras":
    d.loc[t, ["dist_acarreo_plan", "utilizacion_plan", "lluvia_plan"]] = [dist, util, lluvia / 4]
else:
    d.loc[t, "prod_plan"] = prod
X, y, z, ref, ok = B.preparar(d, fam)
prob = art["campeon"][fam][0]
c = Contexto(d, X, z, ref, fam, t, B.indices_entrenamiento(ok, t))
miembros = MIEMBROS_ENSAMBLE if prob == "ens" else (prob,)
qs = []
for m in miembros:
    f = CATALOGO[m][1]
    qs.append(f(c, art["params"][fam].get(m)) if m in ("xgb", "lgbm") else f(c))
q10, q50, q90 = np.mean(qs, axis=0)
qhat = art["familias"][fam]["ultimo_qhat"]
lo, hi = q10 - qhat, q90 + qhat
k = float(C.kappa(q50, hi))
cont = C.contingencia_contrato(k, tipo, meses, tope)
precio = q50 * (1 + cont)
practica = d[fam].iloc[t] * 1.05

u = "S/ por t" if fam == "costo_mac" else "S/ por m³"
es = lambda x, d=2: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", " ")
dec = 0 if fam == "costo_mac" else 2
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Costo esperado P50 ({u})", es(q50))
c2.metric(f"Rango P10–P90 calibrado ({u})", f"{es(lo, dec)} – {es(hi, dec)}")
c3.metric("Contingencia sugerida", f"{es(100 * cont, 1)} %", f"{es(100 * (cont - 0.05), 1)} pp frente al 5 % fijo", delta_color="off")
c4.metric(f"Costo cotizado ({u})", es(precio), f"{es(precio - practica)} frente a la práctica actual", delta_color="off")
st.write(f"Monto total del costo cotizado: **S/ {es(precio * cantidad, 0)}** · Modelo campeón: **{NOMBRES[prob]}** "
         f"(elegido en validación, no en prueba) · Contingencia base κ = {es(100 * k, 1)} % × factor {es(C.FACTOR_CONTRATO[tipo], 1)} × √{meses}")

tab1, tab2, tab3 = st.tabs(["Historia y pronóstico", "Explicación (SHAP)", "Desempeño y monitoreo"])
with tab1:
    fig, ax = plt.subplots(figsize=(9, 3))
    h = df.tail(60)
    ax.plot(h.date, h[fam], color="black", lw=1.2, label="Costo semanal observado")
    fx_ = pd.date_range(df.date.iloc[-1], periods=H + 1, freq="W")[1:]
    ax.fill_between(fx_, lo, hi, color="#143278", alpha=0.18, label="P10–P90 calibrado (próximas 4 semanas)")
    ax.hlines(q50, fx_[0], fx_[-1], color="#143278", lw=2, label="P50")
    ax.hlines(precio, fx_[0], fx_[-1], color="#A84B2F", lw=1.5, ls="--", label="Costo cotizado con contingencia")
    ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
with tab2:
    Xf, yf, zf, rf, okf = B.preparar(df, fam)
    idx = [i for i in range(len(df)) if okf.iloc[i]]
    mx, sv, base = shap_xgb(Xf.iloc[idx], zf.iloc[idx], art["params"][fam]["xgb"])
    import xgboost as xgb
    contrib = mx.get_booster().predict(xgb.DMatrix(X.iloc[[t]]), pred_contribs=True)[0][:-1]
    s = pd.Series(contrib, index=X.columns).sort_values(key=np.abs, ascending=True).tail(8)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(s.index, s.values, color=["#A84B2F" if v > 0 else "#143278" for v in s.values])
    ax.set_title("Contribución de cada variable a esta predicción (XGBoost, escala log)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); st.pyplot(fig)
    st.caption("Rojo: la variable eleva el costo esperado; azul: lo reduce. Valores SHAP exactos (TreeSHAP).")
with tab3:
    t_ = tab[(tab.familia == fam) & (tab.periodo == "prueba")][["nombre", "MAE", "MAPE", "mejora_MAE", "p_Holm", "cob_cal", "winkler"]]
    st.dataframe(t_.rename(columns={"nombre": "Modelo", "mejora_MAE": "Mejora MAE %", "p_Holm": "p (Holm)", "cob_cal": "Cobertura %"}).round(3),
                 hide_index=True, use_container_width=True)
    r = pd.read_csv(RESULTADOS / f"predicciones_{fam}_{prob}.csv", parse_dates=["date"])
    r = r[r.t >= INICIO_TEST]
    cob = 100 * ((r.y >= r.lo) & (r.y <= r.hi)).rolling(12).mean()
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(r.date, cob, color="#143278"); ax.axhline(70, ls="--", color="black", lw=0.7); ax.axhline(80, ls=":", color="grey")
    ax.set_title("Cobertura móvil de 12 semanas (%) · alerta de reentrenamiento bajo 70 %", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); st.pyplot(fig)
