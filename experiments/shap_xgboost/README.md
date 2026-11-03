# SHAP de XGBoost (análisis histórico)

XGBoost fue el modelo elegido hasta el 2026-11-03. Desde la selección por CV
dentro de train ([docs/04 §4.7.2](../../docs/04_modelado.md#472-uso-de-validación-en-la-elección-del-modelo))
el elegido es LightGBM, y `06_evaluation.py` calcula el SHAP de LightGBM.

Este script regenera el SHAP de XGBoost con el procedimiento que usaba `06`
(misma muestra de 3,000 filas de validación), para que las cifras citadas en
E9 y en `docs/02` §2.9 sigan siendo reproducibles.

| Archivo | Contenido |
|---|---|
| `shap_xgboost.py` | El cálculo (segundos) |
| `resultados/shap_importancia_xgboost.csv` | Impacto medio \|SHAP\| y gain nativo por variable |
| `resultados/explicacion_cliente.csv` | Desglose del cliente #22556 (mayor PD de validación) |

```
python experiments/shap_xgboost/shap_xgboost.py
```
