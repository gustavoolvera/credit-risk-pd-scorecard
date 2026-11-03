# Comparación pareada de la CV de 05

Soporte del panel "Diferencias pareadas de AUC" de `05_lightgbm_evaluacion.png`
(2026-11-03). Documentado en
[docs/04_modelado.md §4.7.1](../../docs/04_modelado.md#471-incertidumbre-de-la-comparación-diferencias-pareadas).

`05_train_lightgbm.py` solo guarda la media y la DE (`ddof=0`) de cada modelo.
`comparacion_pareada.py` reconstruye los AUC de cada fold con el mismo
procedimiento, **comprueba que reproducen exactamente** las cifras de
`artifacts/metrics/comparacion_modelos_cv.csv` y calcula las diferencias
pareadas con el IC corregido de Nadeau–Bengio. Los modelos de cada fold existen
solo en memoria: no toca ningún modelo de producción ni nada de `artifacts/`.

| Archivo | Contenido |
|---|---|
| `comparacion_pareada.py` | Reconstrucción, verificación del diseño y diferencias pareadas (≈1 min) |
| `resultados/auc_por_fold.csv` | AUC de los 3 modelos en los 5 folds |
| `resultados/diferencias_pareadas.csv` | Diferencias, IC 95 % corregido y sin corregir, p-valores |

```
python -u experiments/ic_comparacion_cv/comparacion_pareada.py
```
