"""Catálogo de modelos. Cada modelo recibe el contexto de la semana t y devuelve (q10, q50, q90)
en unidades de costo; los modelos puntuales devuelven (nan, q50, nan).

Familias comparadas:
  Referencias ingenuas : último costo, promedio móvil de 4 semanas
  Estadísticos clásicos: suavizamiento exponencial (ETS), SARIMAX con exógenas, ARX (MCO)
  Lineal probabilístico: regresión cuantílica (Koenker y Bassett, 1978)
  Aprendizaje automático: XGBoost y LightGBM con pérdida cuantílica
  Aprendizaje profundo : perceptrón multicapa cuantílico y LSTM cuantílica (PyTorch)
  Modelo fundacional   : Chronos-Bolt (Amazon), sin reentrenamiento (zero-shot)
  Combinación          : promedio de cuantiles de los modelos probabilísticos (ensamble)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .config import H, CUANTILES, SEMILLA
from .features import LINEALES

warnings.filterwarnings("ignore")
NAN3 = lambda q50: (np.nan, float(q50), np.nan)

PARAMS_XGB = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, min_child_weight=3, reg_lambda=2.0)
PARAMS_LGBM = dict(n_estimators=300, num_leaves=4, learning_rate=0.05, min_child_samples=8, subsample=0.8,
                   subsample_freq=1, reg_lambda=2.0)


class Contexto:
    """Información disponible en la semana t para la familia fam."""

    def __init__(self, df, X, z, ref, fam, t, tr, cols=None):
        self.df, self.X, self.z, self.ref, self.fam, self.t, self.tr = df, X, z, ref, fam, t, tr
        self.cols = cols if cols is not None else list(X.columns)

    @property
    def Xtr(self):
        return self.X.iloc[self.tr][self.cols]

    @property
    def ztr(self):
        return self.z.iloc[self.tr]

    @property
    def xt(self):
        return self.X.iloc[[self.t]][self.cols]

    def a_costo(self, zq):
        return self.ref.iloc[self.t] * np.exp(np.asarray(zq, dtype=float))

    @property
    def historia(self):  # costos observados hasta t (inclusive)
        return self.df[self.fam].iloc[: self.t + 1].values


def _ordenar(q):
    q = np.sort(np.asarray(q, dtype=float))
    return float(q[0]), float(q[1]), float(q[2])


# ---------------------------------------------------------------- referencias y clásicos
def ultimo_costo(c):
    return NAN3(c.df[c.fam].iloc[c.t])


def promedio_movil(c):
    return NAN3(c.df[c.fam].iloc[c.t - 3: c.t + 1].mean())


def ets(c):
    h = np.log(c.historia)
    m = SimpleExpSmoothing(h, initialization_method="estimated").fit()
    return NAN3(np.exp(m.forecast(H)).mean())


def sarimax(c):
    """SARIMAX sobre el log del costo. Exógenas: índice Brent x TC (MAC) o planificación de obra (tierras).
    Las exógenas futuras se fijan en su último valor (MAC) o en el plan conocido (tierras)."""
    d, t = c.df, c.t
    y = np.log(d[c.fam].iloc[: t + 1].values)
    if c.fam == "costo_mac":
        ex = np.log(d.brent * d.fx).values.reshape(-1, 1)
        exf = np.repeat(ex[t: t + 1], H, axis=0)
        orden = (1, 1, 0)
    else:
        ex = d[["dist_acarreo", "utilizacion", "lluvia", "diesel"]].values
        exf = ex[t + 1: t + 1 + H].copy()
        exf[:, 3] = ex[t, 3]
        orden = (1, 0, 0)
    m = SARIMAX(y, exog=ex[: t + 1], order=orden, trend="c" if orden[1] == 0 else "n").fit(disp=False)
    f = m.get_forecast(H, exog=exf)
    ci = f.conf_int(alpha=0.2)
    mu = f.predicted_mean
    return _ordenar([np.exp(ci[:, 0].mean()), np.exp(mu).mean(), np.exp(ci[:, 1].mean())])


def _lineales(c):
    return [v for v in LINEALES[c.fam] if v in c.cols]


def arx(c):
    cols = _lineales(c)
    m = sm.OLS(c.ztr, sm.add_constant(c.X.iloc[c.tr][cols])).fit()
    p = m.predict(sm.add_constant(c.X[cols], has_constant="add").iloc[[c.t]]).iloc[0]
    return NAN3(c.a_costo(p))


def regresion_cuantilica(c):
    cols = _lineales(c)
    Xa = sm.add_constant(c.X.iloc[c.tr][cols]); xt = sm.add_constant(c.X[cols], has_constant="add").iloc[[c.t]]
    q = [QuantReg(c.ztr, Xa).fit(q=a, max_iter=5000).predict(xt).iloc[0] for a in CUANTILES]
    return _ordenar(c.a_costo(q))


# ---------------------------------------------------------------- aprendizaje automático
def xgboost_q(c, params=None):
    import xgboost as xgb
    p = dict(PARAMS_XGB, **(params or {}))
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(CUANTILES), random_state=0, **p)
    m.fit(c.Xtr, c.ztr)
    return _ordenar(c.a_costo(m.predict(c.xt)[0]))


def lightgbm_q(c, params=None):
    import lightgbm as lgb
    p = dict(PARAMS_LGBM, **(params or {}))
    q = []
    for a in CUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=0, verbose=-1, **p)
        m.fit(c.Xtr, c.ztr)
        q.append(m.predict(c.xt)[0])
    return _ordenar(c.a_costo(q))


# ---------------------------------------------------------------- aprendizaje profundo
class _Cache:
    """Las redes se reentrenan cada H semanas (costo computacional); entre reentrenamientos se reutilizan."""
    def __init__(self):
        self.d = {}


CACHE = _Cache()


def _pinball_torch(pred, y, taus):
    import torch
    u = y.unsqueeze(1) - pred
    return torch.mean(torch.maximum(taus * u, (taus - 1) * u))


def _entrenar_red(construir_red, Xtr, ytr, semillas=(0, 1, 2), epocas=400, lr=0.01, wd=1e-3):
    import torch
    taus = torch.tensor(CUANTILES, dtype=torch.float32)
    redes = []
    for s in semillas:
        torch.manual_seed(SEMILLA + s)
        net = construir_red()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        n = len(ytr); nval = max(int(0.2 * n), 4)
        Xa, ya, Xv, yv = Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:]  # validación interna = últimas semanas
        mejor, estado, paciencia = np.inf, None, 0
        for _ in range(epocas):
            net.train(); opt.zero_grad()
            loss = _pinball_torch(net(Xa), ya, taus); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                lv = _pinball_torch(net(Xv), yv, taus).item()
            if lv < mejor - 1e-6:
                mejor, estado, paciencia = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                paciencia += 1
                if paciencia > 40:
                    break
        net.load_state_dict(estado)
        redes.append(net.eval())
    return redes


def _estandarizar(Xtr, Xall):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xall - mu) / sd


def mlp_q(c):
    import torch
    nf = len(c.cols)
    clave = ("mlp", c.fam, (c.t - 60) // H)
    zmu, zsd = c.ztr.mean(), c.ztr.std() + 1e-9
    if clave not in CACHE.d:
        Xs, _ = _estandarizar(c.Xtr.values, c.X[c.cols].values)
        Xt = torch.tensor(Xs, dtype=torch.float32); yt = torch.tensor(((c.ztr - zmu) / zsd).values, dtype=torch.float32)
        red = lambda: torch.nn.Sequential(torch.nn.Linear(nf, 16), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                          torch.nn.Linear(16, 3))
        CACHE.d[clave] = (_entrenar_red(red, Xt, yt), c.Xtr.values.mean(0), c.Xtr.values.std(0) + 1e-9, zmu, zsd)
    redes, mu, sd, zmu, zsd = CACHE.d[clave]
    x = torch.tensor((c.xt.values - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        q = np.mean([r(x).numpy()[0] for r in redes], axis=0) * zsd + zmu
    return _ordenar(c.a_costo(q))


class _LSTM:
    def __new__(cls, nf):
        import torch

        class Red(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(nf, 12, batch_first=True)
                self.out = torch.nn.Linear(12, 3)

            def forward(self, x):
                h, _ = self.lstm(x)
                return self.out(h[:, -1, :])
        return Red()


def lstm_q(c, L=8):
    import torch
    nf = len(c.cols)
    clave = ("lstm", c.fam, (c.t - 60) // H)
    Xall = c.X[c.cols].values
    if clave not in CACHE.d:
        mu, sd = c.Xtr.values.mean(0), c.Xtr.values.std(0) + 1e-9
        Xs = (Xall - mu) / sd
        tr = [i for i in c.tr if i - L + 1 >= 0 and not np.isnan(Xs[i - L + 1: i + 1]).any()]
        seq = np.stack([Xs[i - L + 1: i + 1] for i in tr])
        zt = c.z.iloc[tr]; zmu, zsd = zt.mean(), zt.std() + 1e-9
        CACHE.d[clave] = (_entrenar_red(lambda: _LSTM(nf), torch.tensor(seq, dtype=torch.float32),
                                        torch.tensor(((zt - zmu) / zsd).values, dtype=torch.float32), epocas=300),
                          mu, sd, zmu, zsd)
    redes, mu, sd, zmu, zsd = CACHE.d[clave]
    x = torch.tensor(((Xall[c.t - L + 1: c.t + 1] - mu) / sd)[None], dtype=torch.float32)
    with torch.no_grad():
        q = np.mean([r(x).numpy()[0] for r in redes], axis=0) * zsd + zmu
    return _ordenar(c.a_costo(q))


# ---------------------------------------------------------------- modelo fundacional
_CHRONOS = {}


def chronos_q(c, modelo="amazon/chronos-bolt-small"):
    """Predicción zero-shot con Chronos-Bolt sobre la historia del costo (univariada).
    El objetivo es la media de las 4 semanas: se promedian los cuantiles de cada semana (aproximación)."""
    import torch
    from chronos import BaseChronosPipeline
    if modelo not in _CHRONOS:
        _CHRONOS[modelo] = BaseChronosPipeline.from_pretrained(modelo, device_map="cpu", torch_dtype=torch.float32)
    q, _ = _CHRONOS[modelo].predict_quantiles(torch.tensor(c.historia, dtype=torch.float32), prediction_length=H,
                                              quantile_levels=list(CUANTILES))
    return _ordenar(q[0].numpy().mean(0))


CATALOGO = {
    "ultimo": ("Último costo", ultimo_costo, "Referencia"),
    "pm4": ("Promedio móvil (4 sem.)", promedio_movil, "Referencia"),
    "ets": ("Suavizamiento exponencial", ets, "Estadístico"),
    "sarimax": ("SARIMAX con exógenas", sarimax, "Estadístico"),
    "arx": ("ARX (MCO)", arx, "Estadístico"),
    "qr": ("Regresión cuantílica lineal", regresion_cuantilica, "Lineal probabilístico"),
    "xgb": ("XGBoost cuantílico", xgboost_q, "Aprendizaje automático"),
    "lgbm": ("LightGBM cuantílico", lightgbm_q, "Aprendizaje automático"),
    "mlp": ("Red neuronal MLP cuantílica", mlp_q, "Aprendizaje profundo"),
    "lstm": ("LSTM cuantílica", lstm_q, "Aprendizaje profundo"),
    "chronos": ("Chronos-Bolt (zero-shot)", chronos_q, "Modelo fundacional"),
}
MIEMBROS_ENSAMBLE = ("qr", "xgb", "lgbm")
NOMBRES = {k: v[0] for k, v in CATALOGO.items()} | {"ens": "Ensamble (QR + XGB + LGBM)"}
TIPO = {k: v[2] for k, v in CATALOGO.items()} | {"ens": "Combinación"}
