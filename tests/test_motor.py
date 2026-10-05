"""Pruebas automáticas (pytest): ausencia de fuga de información, métricas y calidad de datos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from mcp import datos, features, backtest as B, metricas as M
from mcp.conformal import cqr_aci
from mcp.config import H

DF = datos.generar_sintetico()


@pytest.mark.parametrize("fam", ["costo_mac", "costo_tierras"])
def test_variables_no_usan_futuro(fam):
    """Alterar los datos posteriores a t no cambia las variables de t (salvo la planificación, conocida al cotizar)."""
    t = 90
    d2 = DF.copy()
    for c in ["brent", "fx", "precio_ca", "costo_mac", "costo_tierras", "diesel"]:
        d2.loc[t + 1:, c] = d2.loc[t + 1:, c] * 3
    a, b = features.construir(DF, fam).iloc[: t + 1], features.construir(d2, fam).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_entrenamiento_solo_con_objetivos_observados():
    _, _, _, _, ok = B.preparar(DF, "costo_mac")
    for t in [70, 100, 130]:
        assert max(B.indices_entrenamiento(ok, t)) <= t - H


def test_conformal_sin_fuga():
    r = pd.DataFrame({"y": np.random.default_rng(0).normal(10, 1, 60)})
    r["q10"], r["q90"] = 9.0, 11.0
    r2 = r.copy(); r2.loc[40:, "y"] = 100.0
    a, b = cqr_aci(r), cqr_aci(r2)
    assert np.allclose(a.loc[: 40 + H - 1, ["lo", "hi"]], b.loc[: 40 + H - 1, ["lo", "hi"]])


def test_metricas():
    assert M.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert M.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    y = np.array([1.0, 2.0, 3.0]); assert M.mae(y, y) == 0 and M.mape(y, y + 0.1 * y) == pytest.approx(10)
    assert M.winkler(y, y - 1, y + 1) == pytest.approx(2.0)
    e1, e2 = np.random.default_rng(1).normal(size=(2, 80))
    s1, p1 = M.dm_hln(e1, e2); s2, p2 = M.dm_hln(e2, e1)
    assert s1 == pytest.approx(-s2) and p1 == pytest.approx(p2)
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] >= adj["c"] >= adj["a"]


def test_generador_reproducible_y_esquema():
    d = datos.generar_sintetico(semilla=7)
    assert np.allclose(d.costo_mac, datos.generar_sintetico(semilla=7).costo_mac)
    with pytest.raises(ValueError):
        datos.validar_esquema(pd.concat([d.head(3), d.head(1)]))


def test_falsacion_caminata_sin_ventaja():
    """Con costos que son caminatas aleatorias, el modelo no debe ganar de forma significativa."""
    d = datos.generar_sintetico(semilla=11, escenario="caminata")
    res, _ = B.walk_forward(d, "costo_mac", ["ultimo", "arx"], inicio=84)
    est, p = M.dm_hln((res["arx"].y - res["arx"].q50).values, (res["ultimo"].y - res["ultimo"].q50).values)
    assert not (p < 0.01 and est < 0)
