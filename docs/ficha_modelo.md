# Ficha del modelo (model card)

Formato basado en Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Detalles
- **Nombre:** Motor de costos predictivos, versión 1.0 (prototipo).
- **Tarea:** regresión probabilística del costo unitario medio de las 4 semanas siguientes (P10, P50, P90).
- **Familias:** mezcla asfáltica en caliente (S/ por t) y movimiento de tierras (S/ por m³).
- **Modelos campeones** (elegidos por el puntaje de Winkler en validación):
  - MAC: LightGBM cuantílico.
  - Tierras: regresión cuantílica lineal.
- **Calibración:** CQR + ACI, cobertura nominal del 80 %.
- **Responsable:** Jefatura de Presupuestos, con soporte de un analista de datos.

## Uso previsto
- Apoyar al analista de presupuestos en la cotización: costo esperado, rango y contingencia sugerida según contrato, cliente y plazo.
- El analista puede modificar la sugerencia. Si lo hace, debe registrar el motivo.

## Usos fuera de alcance
- Fijar precios de forma automática sin revisión humana.
- Familias de costo no entrenadas, como puentes o edificaciones.
- Horizontes mayores a 4 semanas sin el factor de plazo.

## Datos
- **Externos reales:**
  - Brent semanal.
  - USD/PEN interbancario (BCRP).
- **Internos:**
  - Prototipo: sintéticos y calibrados con la estructura de costos de la empresa.
  - Producción: historial NIC 2 del ERP.
- **Partición temporal:**
  - Entrenamiento con ventana expansiva.
  - Validación: semanas 61–84.
  - Prueba: semanas 85–138.

## Métricas (periodo de prueba, 54 semanas)
- **Puntuales:** MAE, MAPE, MASE, sesgo.
- **Probabilísticas:** cobertura, prueba de Kupiec, amplitud y puntaje de Winkler.
- **De decisión:**
  - Cobertura del costo real por la cotización.
  - Déficit medio.
  - Contingencia necesaria con igual protección.
- **Significancia:**
  - Prueba DM-HLN con corrección de Holm.
  - IC por bootstrap de bloques.
  - Conjunto de modelos de confianza (MCS).

## Consideraciones éticas y de riesgo
- Los datos de clientes se anonimizan según la Ley N.° 29733.
- **Riesgo de sesgo comercial:** el modelo podría sugerir contingencias bajas que favorezcan ganar licitaciones a costa de pérdidas. Lo mitigan el piso de contingencia (1 %), el tope definido por la gerencia y la revisión humana.
- **Riesgo de deriva:** ante choques de mercado (por ejemplo, el Brent en 2026), lo controlan el PSI, el CUSUM y la cobertura móvil, con alerta de reentrenamiento.

## Limitaciones conocidas
- La serie es corta: 142 semanas. La prueba de 54 semanas solo detecta mejoras del MAE de la MAC de alrededor de 20 % o más, con 80 % de potencia.
- El costo de la MAC se comporta casi como una caminata aleatoria, por lo que la ganancia puntual es pequeña.
- En el prototipo, los resultados dependen de un generador sintético. Las pruebas de robustez cubren 20 réplicas, ruido doble, falsación con caminatas aleatorias y permutación en bloque de las variables explicativas.
