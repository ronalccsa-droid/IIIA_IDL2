"""Carga, validación y generación de datos.

Dos rutas de entrada, con el mismo esquema de salida:
  1. ``cargar_erp``: historial real exportado del ERP (plantilla en data/plantilla_erp.csv).
  2. ``generar_sintetico``: costos internos simulados con la estructura de costos de la empresa,
     sobre variables externas reales (Brent y USD/PEN). Sirve para el prototipo y para las
     pruebas de robustez y de falsación.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DATA, SEMILLA, H

COLUMNAS_ERP = {
    "fecha_semana": "fecha (cierre de semana)",
    "costo_mac": "costo de producción NIC 2 de la MAC, S/ por t",
    "costo_tierras": "costo de producción de movimiento de tierras, S/ por m3",
    "precio_ca": "precio de compra del cemento asfáltico, S/ por t",
    "prod_mac": "toneladas de MAC producidas en la semana",
    "diesel": "precio del diésel, S/ por litro",
    "dist_acarreo": "distancia media de acarreo planificada, km",
    "utilizacion": "utilización planificada de la flota (0-1)",
    "lluvia": "1 si la semana está en temporada de lluvias",
}
RANGOS = {"costo_mac": (100, 1500), "costo_tierras": (1, 60), "precio_ca": (500, 15000), "prod_mac": (0, 50000),
          "diesel": (0.5, 20), "dist_acarreo": (0, 50), "utilizacion": (0, 1), "lluvia": (0, 1),
          "brent": (10, 250), "fx": (2, 6)}


def cargar_externas(ruta=None) -> pd.DataFrame:
    """Brent semanal (US$ por barril) y tipo de cambio interbancario USD/PEN alineado a la misma fecha."""
    ruta = ruta or DATA / "externas_semanal.csv"
    df = pd.read_csv(ruta, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "brent", "fx"]]


def descargar_externas(inicio="2023-07-01", fin="2026-03-27") -> pd.DataFrame:  # pragma: no cover - requiere internet
    """Alternativa para Colab: Brent desde Yahoo Finance (BZ=F) y USD/PEN desde la API del BCRP (PD04637PD)."""
    import yfinance as yf, requests
    b = yf.download("BZ=F", start=inicio, end=fin, interval="1wk", progress=False)["Close"].squeeze()
    b = b.rename("brent").reset_index().rename(columns={"Date": "date"})
    url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04637PD/json/{inicio}/{fin}"
    js = requests.get(url, timeout=30).json()["periods"]
    meses = {"Ene": "01", "Feb": "02", "Mar": "03", "Abr": "04", "May": "05", "Jun": "06", "Jul": "07", "Ago": "08",
             "Sep": "09", "Set": "09", "Oct": "10", "Nov": "11", "Dic": "12"}
    filas = []
    for p in js:
        d, m, a = p["name"].split(".")
        try:
            filas.append((pd.Timestamp(f"20{a}-{meses[m]}-{d}"), float(p["values"][0])))
        except (ValueError, KeyError):
            continue
    fx = pd.DataFrame(filas, columns=["date", "fx"]).sort_values("date")
    return pd.merge_asof(b.sort_values("date"), fx, on="date", direction="backward").dropna()


def validar_esquema(df: pd.DataFrame) -> dict:
    """Controles de calidad del protocolo (sección 7.3). Devuelve un informe; lanza error si hay fallas críticas."""
    req = ["date", "brent", "fx"] + [c for c in COLUMNAS_ERP if c != "fecha_semana"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")
    inf = {"n_semanas": len(df), "fechas_duplicadas": int(df.date.duplicated().sum()),
           "orden_cronologico": bool(df.date.is_monotonic_increasing),
           "saltos_mayores_a_8_dias": int((df.date.diff().dt.days > 8).sum()),
           "faltantes": {c: int(df[c].isna().sum()) for c in req if df[c].isna().any()},
           "fuera_de_rango": {}, "atipicos_3de": {}}
    for c, (lo, hi) in RANGOS.items():
        if c in df:
            k = int(((df[c] < lo) | (df[c] > hi)).sum())
            if k:
                inf["fuera_de_rango"][c] = k
    for c in ["costo_mac", "costo_tierras"]:
        r = np.log(df[c]).diff()
        inf["atipicos_3de"][c] = int((np.abs(r - r.mean()) > 3 * r.std()).sum())
    if inf["fechas_duplicadas"] or not inf["orden_cronologico"]:
        raise ValueError(f"Serie temporal inválida: {inf}")
    return inf


def cargar_erp(ruta, externas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Historial real del ERP con el esquema de data/plantilla_erp.csv."""
    erp = pd.read_csv(ruta, parse_dates=["fecha_semana"]).rename(columns={"fecha_semana": "date"}).sort_values("date")
    ext = externas if externas is not None else cargar_externas()
    df = pd.merge_asof(erp, ext, on="date", direction="backward")
    df["mes"] = df.date.dt.month
    df = construir_objetivos(df)
    validar_esquema(df)
    return df


def generar_sintetico(ext: pd.DataFrame | None = None, semilla: int = SEMILLA, escenario: str = "base",
                      ruido: float = 1.0) -> pd.DataFrame:
    """Costos internos simulados con la estructura de costos de la empresa.

    escenario="base": costo MAC = cemento asfáltico (~53 %) + agregados + diésel + fijo por volumen;
        costo de tierras = costo horario de flota / rendimiento (depende de acarreo, utilización y lluvias).
    escenario="caminata": prueba de falsación. Los costos son caminatas aleatorias independientes de
        todas las variables explicativas; un modelo sin fuga de información no debe superar al último costo.
    ruido: multiplica la desviación de todos los choques (prueba de estrés).
    """
    df = (ext if ext is not None else cargar_externas()).copy().reset_index(drop=True)
    rng = np.random.default_rng(semilla)
    n = len(df)
    target_pc = 13.6 * df.brent * df.fx
    pc = np.empty(n); pc[0] = target_pc.iloc[0]
    for t in range(1, n):
        ref = target_pc.iloc[max(t - 2, 0)]
        pc[t] = pc[t - 1] + 0.35 * (ref - pc[t - 1]) + rng.normal(0, 28 * ruido)
    df["precio_ca"] = pc
    semana = np.arange(n)
    df["mes"] = df.date.dt.month
    df["lluvia"] = df.mes.isin([1, 2, 3]).astype(int)          # temporada de lluvias en Arequipa
    prod = 2900 + 450 * np.sin(2 * np.pi * semana / 52 + 1.2) - 700 * df.lluvia + rng.normal(0, 220 * ruido, n)
    df["prod_mac"] = np.clip(prod, 1400, None)
    df["diesel"] = 0.0095 * df.brent * df.fx + rng.normal(0, 0.06 * ruido, n) + 1.6
    agreg = 108 * (1 + 0.0007 * semana) + rng.normal(0, 1.5 * ruido, n)
    fijo = 45 * 2900 / df.prod_mac
    df["costo_mac"] = 0.056 * df.precio_ca + agreg + 6.0 * df.diesel + fijo + rng.normal(0, 7.5 * ruido, n)
    dist = np.clip(2.5 + np.cumsum(rng.normal(0, 0.25 * ruido, n)) * 0.3
                   + rng.choice([0, 1.5, 3.0], n, p=[.6, .3, .1]), 1.0, 8)
    util = np.clip(0.78 + 0.08 * np.sin(2 * np.pi * semana / 26) - 0.12 * df.lluvia
                   + rng.normal(0, 0.03 * ruido, n), 0.45, 0.95)
    df["dist_acarreo"] = dist
    df["utilizacion"] = util
    rend = 95 * util * (1 - 0.18 * df.lluvia) / (1 + 0.09 * dist)
    costo_hora = 260 + 28 * df.diesel
    df["costo_tierras"] = costo_hora / rend + 2.2 + rng.normal(0, 0.18 * ruido, n)
    if escenario == "caminata":
        for c in ["costo_mac", "costo_tierras"]:
            sd = np.log(df[c]).diff().std()
            df[c] = df[c].iloc[0] * np.exp(np.r_[0, np.cumsum(rng.normal(0, sd, n - 1))])
    return construir_objetivos(df)


def construir_objetivos(df: pd.DataFrame) -> pd.DataFrame:
    """Objetivo: costo medio de las H semanas siguientes. Planificación conocida al cotizar (cronograma de obra)."""
    df = df.copy()
    fut = lambda s: s.shift(-1).rolling(H).mean().shift(-(H - 1))
    for c in ["costo_mac", "costo_tierras"]:
        df[f"y_{c}"] = fut(df[c])
    for c in ["dist_acarreo", "utilizacion", "lluvia"]:
        df[f"{c}_plan"] = fut(df[c])
    df["prod_plan"] = fut(df["prod_mac"])
    return df


def escribir_plantilla(ruta=None):
    """Genera data/plantilla_erp.csv con el esquema que debe exportar el ERP."""
    ruta = ruta or DATA / "plantilla_erp.csv"
    ej = generar_sintetico().head(6)
    out = ej.rename(columns={"date": "fecha_semana"})[list(COLUMNAS_ERP)]
    out.round(4).to_csv(ruta, index=False)
    return ruta
