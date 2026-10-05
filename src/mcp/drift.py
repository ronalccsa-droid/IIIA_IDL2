"""Monitoreo del modelo en producción: deriva de datos (PSI) y deriva de desempeño (cobertura y error)."""
import numpy as np
import pandas as pd


def psi(ref, cur, bins=5):
    """Population Stability Index (Siddiqi, 2006). < 0,10 estable; 0,10-0,25 vigilar; > 0,25 deriva significativa.
    Cortes por quintiles de la referencia y suavizado de Laplace (+0,5) para ventanas cortas."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    a, b = np.histogram(ref, cortes)[0] + 0.5, np.histogram(cur, cortes)[0] + 0.5
    p, q = a / a.sum(), b / b.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def psi_movil(X: pd.DataFrame, fin_ref: int, ventana=26, columnas=None):
    columnas = columnas or list(X.columns)
    ref = X.iloc[:fin_ref].dropna()
    filas = []
    for t in range(fin_ref + ventana, len(X) + 1):
        cur = X.iloc[t - ventana: t].dropna()
        filas.append({"t": t - 1, **{c: psi(ref[c], cur[c]) for c in columnas}})
    return pd.DataFrame(filas).set_index("t")


def monitor_desempeno(r: pd.DataFrame, errores_ref=None, ventana=12, cob_min=70.0, k=0.5, h=5.0):
    """Cobertura móvil del intervalo y CUSUM unilateral (Page, 1954) del error absoluto estandarizado.
    La media y la desviación de referencia salen del periodo de validación (errores_ref).
    Alerta de reentrenamiento si la cobertura móvil cae bajo cob_min o el CUSUM supera h; el CUSUM se
    reinicia después de cada alarma."""
    m = r.copy()
    m["dentro"] = ((m.y >= m.lo) & (m.y <= m.hi)).astype(float)
    m["cob_movil"] = 100 * m.dentro.rolling(ventana, min_periods=ventana).mean()
    e = np.abs(m.y - m.q50)
    ref = np.abs(np.asarray(errores_ref)) if errores_ref is not None else e.iloc[:ventana].values
    mu, sd = ref.mean(), ref.std() + 1e-9
    s, cus, alarma = 0.0, [], []
    for v in (e - mu) / sd:
        s = max(0.0, s + v - k)
        cus.append(s); alarma.append(s > h)
        if s > h:
            s = 0.0
    m["cusum"] = cus
    m["alarma_cusum"] = alarma
    m["alarma_cobertura"] = m.cob_movil < cob_min
    m["alerta"] = m.alarma_cusum | m.alarma_cobertura
    return m
