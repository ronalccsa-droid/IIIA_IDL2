"""Validación walk-forward con ventana expansiva, ajuste de hiperparámetros y selección de campeón.

Regla de no fuga: en la semana t solo se entrena con objetivos ya observados, es decir, con semanas
i <= t - H (el objetivo de i es la media de i+1..i+H).
Periodos: validación [INICIO_VAL, INICIO_TEST) para elegir hiperparámetros y campeón;
prueba [INICIO_TEST, fin] solo para reportar resultados.
"""
from __future__ import annotations
import itertools, time
import numpy as np
import pandas as pd
from .config import H, INICIO_VAL, INICIO_TEST, CUANTILES
from .features import construir, referencia
from .modelos import Contexto, CATALOGO, MIEMBROS_ENSAMBLE, xgboost_q, lightgbm_q, CACHE
from .conformal import cqr_aci
from . import metricas as M


def preparar(df, fam):
    X = construir(df, fam); y = df[f"y_{fam}"]; ref = referencia(df, fam)
    ok = X.notna().all(axis=1) & y.notna()
    z = np.log(y / ref)
    return X, y, z, ref, ok


def indices_entrenamiento(ok, t):
    return [i for i in range(t - H + 1) if ok.iloc[i]]


def walk_forward(df, fam, modelos, params=None, inicio=INICIO_VAL, paso=1, cols=None, ventana=None, verbose=False):
    """Devuelve {modelo: DataFrame(t, date, y, q10, q50, q90)} para t >= inicio."""
    params = params or {}
    X, y, z, ref, ok = preparar(df, fam)
    CACHE.d.clear()
    out = {m: [] for m in modelos}
    tiempos = {m: 0.0 for m in modelos}
    for t in range(inicio, len(df), paso):
        if not ok.iloc[t]:
            continue
        tr = indices_entrenamiento(ok, t)
        if ventana:
            tr = tr[-ventana:]
        c = Contexto(df, X, z, ref, fam, t, tr, cols)
        for m in modelos:
            t0 = time.perf_counter()
            f = CATALOGO[m][1]
            q = f(c, params.get(m)) if m in ("xgb", "lgbm") else f(c)
            tiempos[m] += time.perf_counter() - t0
            out[m].append(dict(t=t, date=df.date.iloc[t], y=y.iloc[t], q10=q[0], q50=q[1], q90=q[2]))
        if verbose and t % 10 == 0:
            print(fam, t, end=" | ", flush=True)
    res = {m: pd.DataFrame(v).set_index("t") for m, v in out.items()}
    if all(k in res for k in MIEMBROS_ENSAMBLE):
        e = res[MIEMBROS_ENSAMBLE[0]][["date", "y"]].copy()
        for q in ("q10", "q50", "q90"):
            e[q] = np.mean([res[k][q].values for k in MIEMBROS_ENSAMBLE], axis=0)
        res["ens"] = e
    return res, tiempos


def ajustar_hiperparametros(df, fam, paso=2):
    """Búsqueda en rejilla con origen móvil SOLO en el periodo de validación. Criterio: pérdida cuantílica media."""
    X, y, z, ref, ok = preparar(df, fam)
    origenes = [t for t in range(INICIO_VAL, INICIO_TEST, paso) if ok.iloc[t]]
    rejillas = {
        "xgb": [dict(max_depth=d, learning_rate=lr, n_estimators=ne)
                for d, lr, ne in itertools.product([2, 3, 4], [0.03, 0.05, 0.1], [150, 250, 400])],
        "lgbm": [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
                 for nl, lr, ne, mc in itertools.product([4, 8], [0.03, 0.05, 0.1], [150, 300], [5, 10])],
    }
    fun = {"xgb": xgboost_q, "lgbm": lightgbm_q}
    mejor, tabla = {}, []
    for m, rej in rejillas.items():
        for p in rej:
            perd = []
            for t in origenes:
                c = Contexto(df, X, z, ref, fam, t, indices_entrenamiento(ok, t))
                q = fun[m](c, p)
                perd.append(np.mean([M.pinball([y.iloc[t]], [qq], a) for qq, a in zip(q, CUANTILES)]))
            tabla.append(dict(modelo=m, **p, pinball=float(np.mean(perd))))
        sub = [r for r in tabla if r["modelo"] == m]
        b = min(sub, key=lambda r: r["pinball"])
        mejor[m] = {k: v for k, v in b.items() if k not in ("modelo", "pinball")}
    return mejor, pd.DataFrame(tabla)


def evaluar(res, fam, base="ultimo"):
    """Métricas en validación y en prueba para todos los modelos; intervalos calibrados con CQR + ACI."""
    filas, cal = [], {}
    for m, r in res.items():
        tiene_int = r.q10.notna().all()
        if tiene_int:
            r = cqr_aci(r)
        cal[m] = r
    b = cal[base]
    for per, cond in (("validacion", lambda r: r.index < INICIO_TEST), ("prueba", lambda r: r.index >= INICIO_TEST)):
        rb = b[cond(b)]
        pvals = {}
        for m, r in cal.items():
            rr = r[cond(r)]; y = rr.y.values; p = rr.q50.values
            f = dict(familia=fam, periodo=per, modelo=m, n=len(rr), MAE=M.mae(y, p), RMSE=M.rmse(y, p), MAPE=M.mape(y, p),
                     MASE=M.mase(y, p, rb.q50.values), sesgo=M.sesgo(y, p),
                     mejora_MAE=100 * (1 - M.mae(y, p) / M.mae(y, rb.q50.values)))
            if m != base:
                f["DM"], f["p_DM"] = M.dm_hln(y - p, y - rb.q50.values)
                pvals[m] = f["p_DM"]
                if per == "prueba":
                    f["IC95_inf"], f["IC95_sup"] = M.bootstrap_mejora(y, p, rb.q50.values)
            if "lo" in rr:
                f.update(cob_bruta=M.cobertura(y, rr.q10, rr.q90), cob_cal=M.cobertura(y, rr.lo, rr.hi),
                         kupiec_p=M.kupiec(y, rr.lo, rr.hi), amplitud_rel=float(100 * np.mean((rr.hi - rr.lo) / rr.q50)),
                         winkler=M.winkler(y, rr.lo, rr.hi),
                         pinball=float(np.mean([M.pinball(y, rr[q], a) for q, a in zip(("q10", "q50", "q90"), CUANTILES)])))
            filas.append(f)
        adj = M.holm(pvals)
        for f in filas:
            if f["periodo"] == per and f["familia"] == fam and f["modelo"] in adj:
                f["p_Holm"] = adj[f["modelo"]]
    return pd.DataFrame(filas), cal


def elegir_campeon(tabla, fam):
    """Campeón probabilístico = menor puntaje de Winkler en VALIDACIÓN (nunca en prueba).
    Campeón puntual = menor MAE en validación."""
    v = tabla[(tabla.familia == fam) & (tabla.periodo == "validacion")]
    prob = v[v.winkler.notna()].sort_values("winkler").iloc[0].modelo
    punt = v.sort_values("MAE").iloc[0].modelo
    return prob, punt
