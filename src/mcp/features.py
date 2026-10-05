"""Ingeniería de variables. Todas usan solo información disponible en la semana t."""
import numpy as np
import pandas as pd

# Grupos para el estudio de ablación
GRUPOS = {
    "costo_mac": {
        "Mercado externo (Brent, TC)": ["brent", "fx", "idx", "r_brent4", "r_fx4"],
        "Insumo (cemento asfáltico)": ["precio_ca", "r_ca4"],
        "Planificación (producción)": ["prod_plan"],
        "Estacionalidad": ["mes_sin", "mes_cos"],
        "Autorregresivas (rezagos)": ["lag1", "ma4"],
    },
    "costo_tierras": {
        "Mercado externo (Brent, TC)": ["brent", "fx", "idx", "r_brent4", "r_fx4"],
        "Insumo (diésel)": ["diesel"],
        "Planificación (acarreo, utilización, lluvias)": ["dist_plan", "util_plan", "lluvia_plan"],
        "Estacionalidad": ["mes_sin", "mes_cos"],
        "Autorregresivas (rezagos)": ["lag1", "ma4"],
    },
}
# Variables del modelo lineal (ARX y regresión cuantílica)
LINEALES = {"costo_mac": ["r_brent4", "r_fx4", "mes_sin", "mes_cos", "r_ca4"],
            "costo_tierras": ["r_brent4", "r_fx4", "mes_sin", "mes_cos", "dist_plan", "util_plan", "lluvia_plan", "diesel"]}


def construir(d: pd.DataFrame, fam: str) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["brent"] = d.brent; X["fx"] = d.fx; X["idx"] = d.brent * d.fx
    X["r_brent4"] = np.log(d.brent).diff(4); X["r_fx4"] = np.log(d.fx).diff(4)
    X["mes_sin"] = np.sin(2 * np.pi * d.mes / 12); X["mes_cos"] = np.cos(2 * np.pi * d.mes / 12)
    if fam == "costo_mac":
        X["precio_ca"] = d.precio_ca; X["r_ca4"] = np.log(d.precio_ca).diff(4)
        X["prod_plan"] = d.prod_plan
        X["lag1"] = d.costo_mac; X["ma4"] = d.costo_mac.rolling(4).mean()
    else:
        X["dist_plan"] = d.dist_acarreo_plan; X["util_plan"] = d.utilizacion_plan; X["lluvia_plan"] = d.lluvia_plan
        X["diesel"] = d.diesel
        X["lag1"] = d.costo_tierras; X["ma4"] = d.costo_tierras.rolling(4).mean()
    return X


def referencia(d: pd.DataFrame, fam: str) -> pd.Series:
    """Escala del objetivo: la MAC se modela como log-ratio frente al último costo (serie I(1));
    tierras, en log-nivel (serie que revierte a la media según la planificación)."""
    return d[fam] if fam == "costo_mac" else pd.Series(1.0, index=d.index)
