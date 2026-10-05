"""Calibración conformal de intervalos para series de tiempo.

CQR (Romano et al., 2019): el puntaje de no conformidad es s = max(q10 - y, y - q90).
ACI (Gibbs y Candès, 2021): el nivel alpha_t se ajusta en línea según los fallos de cobertura.
El puntaje de la semana t solo se conoce cuando se observa y_t, es decir, H semanas después;
por eso se usan únicamente los puntajes con t' <= t - H (sin fuga de información).
"""
import numpy as np
import pandas as pd
from .config import ALPHA, GAMMA_ACI, H, VENTANA_CONFORMAL


def cqr_aci(r: pd.DataFrame, alpha=ALPHA, gamma=GAMMA_ACI, ventana=VENTANA_CONFORMAL, h=H, minimo=8):
    """r: DataFrame indexado por t con columnas y, q10, q90. Devuelve lo, hi, alpha_t, qhat."""
    a_t = alpha
    puntajes, lo, hi, al, qh = [], [], [], [], []
    pend = []  # errores de cobertura pendientes de revelar
    for t in r.index:
        # revelar resultados de semanas cuyo objetivo ya se observó
        for (tt, s, err) in [p for p in pend if p[0] <= t - h]:
            puntajes.append(s)
            a_t = a_t + gamma * (alpha - err)
        pend = [p for p in pend if p[0] > t - h]
        disp = puntajes[-ventana:]
        if len(disp) >= minimo:
            nivel = min(max(1 - a_t, 0.0), 1.0)
            qhat = float(np.quantile(disp, nivel)) if nivel < 1 else float(max(disp))
        else:
            qhat = 0.0
        l, u = r.at[t, "q10"] - qhat, r.at[t, "q90"] + qhat
        lo.append(l); hi.append(u); al.append(a_t); qh.append(qhat)
        y = r.at[t, "y"]
        pend.append((t, max(r.at[t, "q10"] - y, y - r.at[t, "q90"]), float(not (l <= y <= u))))
    out = r.copy()
    out["lo"], out["hi"], out["alpha_t"], out["qhat"] = lo, hi, al, qh
    return out
