# Motor de costos predictivos

Predicción probabilística a cuatro semanas del costo unitario de la mezcla asfáltica en caliente (MAC) y del
movimiento de tierras, y conversión del intervalo calibrado en una contingencia diferenciada por contrato.
Proyecto de Maestría en Ciencia de Datos e Inteligencia Artificial, caso INDUMAQ S.R.L. (Arequipa, Perú).

## Estructura

```
motor-costos-predictivos/
├── data/
│   ├── externas_semanal.csv     Brent semanal y USD/PEN interbancario (datos reales, 2023-07 a 2026-03)
│   └── plantilla_erp.csv        esquema que debe exportar el ERP para usar datos reales
├── src/mcp/                     paquete del motor
│   ├── config.py                parámetros fijados antes de mirar la prueba
│   ├── datos.py                 carga, validación de calidad y generador sintético calibrado
│   ├── features.py              ingeniería de variables (solo información disponible en t)
│   ├── modelos.py               12 modelos: referencias, ETS, SARIMAX, ARX, QR, XGBoost, LightGBM, MLP, LSTM, Chronos, ensamble
│   ├── backtest.py              walk-forward, ajuste de hiperparámetros y selección del campeón
│   ├── conformal.py             CQR + ACI sin fuga de información
│   ├── metricas.py              MAE, MAPE, MASE, pinball, Winkler, Kupiec, DM-HLN, Holm, bootstrap, MCS, potencia
│   ├── contingencia.py          política de contingencia y backtest frente a la práctica actual
│   ├── explicabilidad.py        SHAP (TreeSHAP) e importancia por permutación
│   ├── drift.py                 PSI y CUSUM para el monitoreo en producción
│   └── economico.py             VAN, TIR, B/C, escenarios, tornado y Monte Carlo
├── scripts/
│   ├── ejecutar_todo.py         experimento completo → resultados/ y artefactos/
│   ├── robustez.py              Réplicas del generador, ruido doble y prueba de falsación
│   ├── permutacion.py           Permutación en bloque de las variables explicativas
│   └── construir_notebook.py    genera el cuaderno de Colab autocontenido
├── notebooks/motor_costos_predictivos_colab.ipynb
├── app/tablero.py               tablero Streamlit para el analista de presupuestos
├── tests/test_motor.py          pruebas automáticas (fuga de información, métricas, calidad de datos)
├── docs/ficha_modelo.md         ficha del modelo (model card)
└── resultados/                  métricas, predicciones, figuras y registros de ejecución
```

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 7 pruebas
python scripts/ejecutar_todo.py             # unos 5 minutos en CPU
python scripts/robustez.py 20               # unos 25 minutos
python scripts/permutacion.py 20            # unos 3 minutos
streamlit run app/tablero.py                # tablero en http://localhost:8501
```

En Google Colab: abrir `notebooks/motor_costos_predictivos_colab.ipynb` y ejecutar todo. El cuaderno contiene el
paquete y los datos, por lo que no requiere clonar el repositorio.

## Usar los datos reales de la empresa

1. Exportar del ERP un CSV semanal con las columnas de `data/plantilla_erp.csv`.
2. `python scripts/ejecutar_todo.py --erp ruta/al/archivo.csv`
3. `validar_esquema` detiene la ejecución si hay fechas duplicadas o desordenadas e informa faltantes, valores fuera de rango y atípicos.

## Decisiones de diseño que protegen la validez

- Ningún parámetro, hiperparámetro ni modelo campeón se eligió mirando el periodo de prueba (semanas 85 a 138).
- En la semana t solo se entrena con objetivos observados (t − 4 o antes); las pruebas automáticas lo verifican.
- La calibración conformal solo usa puntajes cuyo valor real ya se conoce.
- La prueba de falsación (costos que son caminatas aleatorias) y la permutación de las variables explicativas confirman que no hay fuga de información.
- Las comparaciones múltiples se corrigen con Holm y se complementan con el conjunto de modelos de confianza (MCS).

## Limitación principal

Los costos internos del prototipo son sintéticos y calibrados con la estructura de costos de la empresa; las
variables externas son reales. Las conclusiones sobre el método son válidas; las cifras de mejora deben
recalcularse con el historial del ERP antes de usarse en decisiones comerciales.
