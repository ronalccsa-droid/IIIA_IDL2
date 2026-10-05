"""Métricas de error, pruebas de significancia e intervalos de confianza."""
import numpy as np
import pandas as pd
from scipy import stats
from .config import H, ALPHA


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mape(y, p): y = np.asarray(y); return float(100 * np.mean(np.abs(y - np.asarray(p)) / y))
def sesgo(y, p): return float(np.mean(np.asarray(y) - np.asarray(p)))


def mase(y, p, y_naive):
    """MAE escalado por el MAE del último costo en el mismo periodo (Hyndman y Koehler, 2006)."""
    return mae(y, p) / mae(y, y_naive)


def pinball(y, q, a):
    u = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(a * u, (a - 1) * u)))


def winkler(y, lo, hi, alpha=ALPHA):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)))


def cobertura(y, lo, hi):
    y = np.asarray(y); return float(100 * np.mean((y >= lo) & (y <= hi)))


def kupiec(y, lo, hi, nominal=1 - ALPHA):
    """Prueba de cobertura incondicional de Kupiec (1995). H0: cobertura = nominal."""
    y = np.asarray(y); x = int(np.sum((y < lo) | (y > hi))); n = len(y); p = 1 - nominal
    ph = x / n
    if ph in (0, 1):
        return float("nan")
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(1 - stats.chi2.cdf(lr, 1))


def dm_hln(e1, e2, h=H, potencia=1):
    """Diebold y Mariano (1995) con la corrección de Harvey, Leybourne y Newbold (1997).
    Estadístico negativo: el modelo 1 tiene menor pérdida que el 2. Devuelve (estadístico, p bilateral)."""
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    T = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    if var <= 0:
        var = gam[0] / T
    est = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(est), float(2 * (1 - stats.t.cdf(abs(est), df=T - 1)))


def holm(pvals: dict) -> dict:
    """Corrección de Holm-Bonferroni por comparaciones múltiples."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, acc = {}, 0.0
    for i, (k, p) in enumerate(items):
        acc = max(acc, min(1.0, (m - i) * p)); out[k] = acc
    return out


def bootstrap_mejora(y, p_modelo, p_base, bloque=H, B=2000, semilla=0):
    """IC 95 % de la mejora relativa del MAE por bootstrap de bloques móviles (Künsch, 1989)."""
    rng = np.random.default_rng(semilla)
    a = np.abs(np.asarray(y) - np.asarray(p_modelo)); b = np.abs(np.asarray(y) - np.asarray(p_base))
    n = len(a); k = int(np.ceil(n / bloque)); res = []
    for _ in range(B):
        ini = rng.integers(0, n - bloque + 1, k)
        idx = np.concatenate([np.arange(s, s + bloque) for s in ini])[:n]
        res.append(100 * (1 - a[idx].mean() / b[idx].mean()))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def mcs(perdidas: pd.DataFrame, size=0.10, bloque=H, reps=2000, semilla=0):
    """Conjunto de modelos de confianza de Hansen, Lunde y Nason (2011): modelos que no son
    estadísticamente peores que el mejor al nivel 'size'. Devuelve {modelo: p-valor MCS}."""
    from arch.bootstrap import MCS
    m = MCS(perdidas, size=size, block_size=bloque, method="R", reps=reps, seed=semilla)
    m.compute()
    return m.pvalues.iloc[:, 0].to_dict()


def efecto_minimo_detectable(e_modelo, e_base, h=H, alfa=0.05, potencia=0.80):
    """Mejora mínima del MAE (en %) detectable con la prueba DM dado el tamaño y la varianza de la muestra."""
    d = np.abs(e_modelo) - np.abs(e_base); T = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    se = np.sqrt(max(gam[0] + 2 * sum(gam[1:]), gam[0]) / T)
    z = stats.t.ppf(1 - alfa / 2, T - 1) + stats.norm.ppf(potencia)
    return float(100 * z * se / np.mean(np.abs(e_base)))
