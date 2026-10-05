"""Parámetros globales del experimento.

Todos los valores se fijan antes de mirar el periodo de prueba y no se modifican después.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RESULTADOS / "figuras"
ARTEFACTOS = RAIZ / "artefactos"

SEMILLA = 2026          # semilla del generador sintético y de los modelos
H = 4                   # horizonte de predicción (semanas)
INICIO_VAL = 60         # primera semana (índice 0) con predicción fuera de muestra: validación
INICIO_TEST = 84        # primera semana del periodo de prueba (no se usa para ninguna decisión)
CUANTILES = (0.1, 0.5, 0.9)
ALPHA = 0.20            # 1 - cobertura nominal del intervalo P10-P90
GAMMA_ACI = 0.03        # tasa de aprendizaje de la calibración conformal adaptativa
VENTANA_CONFORMAL = 40  # puntajes de no conformidad usados por la calibración
CONTINGENCIA_BASE = 0.05  # práctica actual: contingencia fija por criterio
PISO_CONTINGENCIA = 0.01
FAMILIAS = ("costo_mac", "costo_tierras")
NOMBRE_FAMILIA = {"costo_mac": "MAC (S/ por t)", "costo_tierras": "Movimiento de tierras (S/ por m³)"}

for _p in (RESULTADOS, FIGURAS, ARTEFACTOS):
    _p.mkdir(parents=True, exist_ok=True)
