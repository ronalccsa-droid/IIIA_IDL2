"""Explicabilidad: valores SHAP exactos para árboles (Lundberg y Lee, 2017) calculados con XGBoost
(pred_contribs=True, algoritmo TreeSHAP) e importancia por permutación agnóstica al modelo."""
import numpy as np
import pandas as pd
import xgboost as xgb
from .config import CUANTILES
from .modelos import PARAMS_XGB


def shap_xgb(X: pd.DataFrame, z: pd.Series, params=None, cuantil=0.5):
    p = dict(PARAMS_XGB, **(params or {}))
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=cuantil, random_state=0, **p).fit(X, z)
    contrib = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    sv = pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)
    base = float(contrib[0, -1])
    return m, sv, base


def importancia_permutacion(predecir, X, y, metrica, repeticiones=30, semilla=0):
    rng = np.random.default_rng(semilla)
    base = metrica(y, predecir(X)); out = {}
    for c in X.columns:
        d = []
        for _ in range(repeticiones):
            Xp = X.copy(); Xp[c] = rng.permutation(Xp[c].values)
            d.append(metrica(y, predecir(Xp)) - base)
        out[c] = float(np.mean(d))
    return pd.Series(out).sort_values(ascending=False)
