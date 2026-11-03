# Selección del modelo sin usar validación (A1)

Documentado en
[docs/04_modelado.md §4.7.2](../../docs/04_modelado.md#472-uso-de-validación-en-la-elección-del-modelo).

**Pregunta.** La regla de `06` elige el modelo con métricas del conjunto de
validación. ¿Qué elegiría la misma regla con métricas de una CV dentro de train?

**Regla (fijada antes de ejecutar).** Entre los modelos con AUC medio a menos de
0.002 del mejor, el de menor error de calibración medio. Idéntica a la de `06`
salvo por el origen de las métricas.

**Resultado.** La regla vigente elige XGBoost y la regla CV elige LightGBM.
Ninguna diferencia entre ambos boosters es significativa (AUC, calibración,
Brier, beneficio).

| Archivo | Contenido |
|---|---|
| `seleccion_cv.py` | CV 5×5 sobre train para los 3 modelos (~3 min). No modifica el pipeline |
| `resultados/cv_por_particion.csv` | Métricas de cada modelo en cada partición |
| `resultados/resumen_cv.csv` | Media y desviación por modelo |
| `resultados/diferencias_pareadas.csv` | Diferencias pareadas con IC 95 % de Nadeau–Bengio |
| `resultados/eleccion.json` | Modelo que elige cada regla (CV en train y validación) |

```
python -u experiments/seleccion_cv/seleccion_cv.py
```
