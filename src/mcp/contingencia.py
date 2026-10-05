"""Política de contingencia y su evaluación retrospectiva frente a la práctica actual."""
import numpy as np
import pandas as pd
from .config import CONTINGENCIA_BASE, PISO_CONTINGENCIA

# Matriz de contingencia (tabla 9 del documento): factor según quién asume el riesgo de precio
FACTOR_CONTRATO = {
    "Entidad pública con fórmula polinómica": 0.4,
    "Privado nacional con reajuste pactado": 0.7,
    "Suma alzada sin reajuste": 1.0,
    "Suma alzada con cliente de bajo reconocimiento de adicionales": 1.2,
}


def kappa(q50, hi):
    """Contingencia base de precio: distancia relativa entre P90 calibrado y P50, con piso."""
    return np.maximum((np.asarray(hi) - np.asarray(q50)) / np.asarray(q50), PISO_CONTINGENCIA)


def contingencia_contrato(k, tipo, meses=1, tope=0.25):
    """kappa x factor contractual x raíz del plazo (meses), con tope definido por la gerencia."""
    return float(min(k * FACTOR_CONTRATO[tipo] * np.sqrt(max(meses, 1)), tope))


def backtest(te: pd.DataFrame, naive: pd.Series, base=CONTINGENCIA_BASE):
    """Compara 'último costo + 5 %' con 'P50 + kappa' en el periodo de prueba."""
    y = te.y.values; nv = naive.loc[te.index].values
    cot_b = nv * (1 + base)
    k = kappa(te.q50.values, te.hi.values); cot_m = te.q50.values * (1 + k)
    out = dict(cont_media_base=100 * base, cont_media_motor=100 * k.mean(), cont_min_motor=100 * k.min(),
               cont_max_motor=100 * k.max(), cobertura_base=100 * np.mean(cot_b >= y), cobertura_motor=100 * np.mean(cot_m >= y),
               deficit_medio_base=float(np.mean(np.maximum(y - cot_b, 0))), deficit_medio_motor=float(np.mean(np.maximum(y - cot_m, 0))),
               exceso_medio_base=float(np.mean(np.maximum(cot_b - y, 0))), exceso_medio_motor=float(np.mean(np.maximum(cot_m - y, 0))),
               costo_medio=float(np.mean(y)))
    # Igual protección: contingencia necesaria para cubrir el costo real en 90 % de las semanas
    need_b = np.quantile(y / nv - 1, 0.90)
    esc = next(s for s in np.linspace(0.2, 3.0, 281) if np.mean(te.q50.values * (1 + s * k) >= y) >= 0.90)
    out.update(cont_base90=100 * float(need_b), cont_motor90=100 * float((esc * k).mean()),
               precio_base90=float(np.mean(nv * (1 + need_b) / y)), precio_motor90=float(np.mean(te.q50.values * (1 + esc * k) / y)))
    out["ventaja_precio_pp"] = 100 * (out["precio_base90"] - out["precio_motor90"])
    return out
