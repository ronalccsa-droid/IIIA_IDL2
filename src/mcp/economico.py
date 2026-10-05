"""Evaluación económica incremental (con proyecto vs. sin proyecto) a tres años."""
import numpy as np

SUPUESTOS = dict(vol_mac=150_000, vol_t_cot=3_000_000, tasa_exito=0.30, share_sa=0.40, margen_contrib=0.12,
                 captura=0.30, elasticidad=1.0, cot_mes=30, horas_ahorro=3.0, costo_hora=45.0, tasa=0.12, anios=3)
INVERSION = {"Analista / científico de datos (6 meses a tiempo completo)": 54_000,
             "Desarrollo e integración en el ERP y tablero": 18_000,
             "Asesoría estadística externa": 12_000,
             "Capacitación y gestión del cambio": 6_000}
INVERSION["Imprevistos (10 %)"] = 0.10 * sum(INVERSION.values())
OPEX = {"Infraestructura en la nube y cómputo": 6_000, "Mantenimiento y reentrenamiento (0,25 FTE)": 27_000,
        "Monitoreo y auditoría del modelo": 4_000}


def flujo(cont, s, inv_mult=1.0, opex_mult=1.0):
    """cont: resultados de contingencia.backtest por familia."""
    ct, cm = cont["costo_tierras"], cont["costo_mac"]
    vol_t = s["vol_t_cot"] * s["tasa_exito"]
    b1 = (max(ct["deficit_medio_base"] - ct["deficit_medio_motor"], 0) * vol_t
          + max(cm["deficit_medio_base"] - cm["deficit_medio_motor"], 0) * s["vol_mac"]) * s["share_sa"]
    dprecio = max(ct["ventaja_precio_pp"], 0) * s["captura"]
    dexito = min(dprecio * s["elasticidad"] / 100, 0.10)
    b2 = s["vol_t_cot"] * dexito * ct["costo_medio"] * (1 + s["margen_contrib"]) * s["margen_contrib"]
    b3 = s["cot_mes"] * 12 * s["horas_ahorro"] * s["costo_hora"]
    ben = dict(perdidas_evitadas=b1, competitividad=b2, horas=b3)
    inv = sum(INVERSION.values()) * inv_mult; op = sum(OPEX.values()) * opex_mult
    rampa = [0.5] + [1.0] * (s["anios"] - 1)
    fl = [-inv] + [sum(ben.values()) * rampa[i] - op for i in range(s["anios"])]
    return ben, inv, op, fl


def van(fl, r): return float(sum(f / (1 + r) ** i for i, f in enumerate(fl)))


def tir(fl):
    lo, hi = -0.99, 10.0
    if van(fl, lo + 1e-9) * van(fl, hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if van(fl, mid) > 0 else (lo, mid)
    return mid


def recuperacion(fl, r):
    acc = 0.0
    for i, f in enumerate(fl):
        prev = acc; acc += f / (1 + r) ** i
        if acc >= 0 and i > 0:
            return i - 1 + (-prev) / (f / (1 + r) ** i)
    return float("nan")


def evaluar(cont, s=None, **kw):
    s = s or SUPUESTOS
    ben, inv, op, fl = flujo(cont, s, **kw)
    rampa = [0.5] + [1.0] * (s["anios"] - 1)
    vb = sum(sum(ben.values()) * rampa[i - 1] / (1 + s["tasa"]) ** i for i in range(1, s["anios"] + 1))
    vc = inv + sum(op / (1 + s["tasa"]) ** i for i in range(1, s["anios"] + 1))
    return dict(beneficios=ben, ben_total=sum(ben.values()), inversion=inv, opex=op, flujo=fl,
                VAN=van(fl, s["tasa"]), TIR=tir(fl), BC=vb / vc, recuperacion=recuperacion(fl, s["tasa"]))


def escenarios(cont):
    S = SUPUESTOS
    return {"Pesimista": evaluar(cont, dict(S, captura=0.10, elasticidad=0.5, tasa_exito=0.25, horas_ahorro=1.5), inv_mult=1.3, opex_mult=1.3),
            "Base": evaluar(cont), "Optimista": evaluar(cont, dict(S, captura=0.50, elasticidad=1.5, horas_ahorro=4.0)),
            "Sin competitividad": evaluar(cont, dict(S, captura=0.0))}


def tornado(cont):
    S = SUPUESTOS; out = {}
    for k, lab in [("captura", "Captura de la ventaja de precio"), ("elasticidad", "Elasticidad de adjudicación"),
                   ("margen_contrib", "Margen de contribución"), ("vol_t_cot", "Volumen cotizado de tierras"),
                   ("share_sa", "Proporción a suma alzada"), ("horas_ahorro", "Horas ahorradas por cotización")]:
        out[lab] = (evaluar(cont, dict(S, **{k: S[k] * 0.5}))["VAN"], evaluar(cont, dict(S, **{k: S[k] * 1.5}))["VAN"])
    out["Inversión y costos operativos"] = (evaluar(cont, inv_mult=1.5, opex_mult=1.5)["VAN"], evaluar(cont, inv_mult=0.5, opex_mult=0.5)["VAN"])
    return out


def montecarlo(cont, n=10_000, semilla=0):
    """Simulación de Monte Carlo con distribuciones uniformes sobre los supuestos inciertos."""
    rng = np.random.default_rng(semilla); vans = []
    for _ in range(n):
        s = dict(SUPUESTOS, captura=rng.uniform(0.0, 0.5), elasticidad=rng.uniform(0.5, 1.5), tasa_exito=rng.uniform(0.25, 0.35),
                 share_sa=rng.uniform(0.3, 0.5), margen_contrib=rng.uniform(0.08, 0.16), vol_t_cot=rng.uniform(2e6, 4e6),
                 horas_ahorro=rng.uniform(1.5, 4.0))
        vans.append(evaluar(cont, s, inv_mult=rng.uniform(1.0, 1.3), opex_mult=rng.uniform(1.0, 1.3))["VAN"])
    v = np.array(vans)
    return dict(prob_van_pos=float(100 * np.mean(v > 0)), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), media=float(v.mean())), v
