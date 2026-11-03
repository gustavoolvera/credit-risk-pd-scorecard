# Ablación del bloque de atrasos en XGBoost

Experimento aislado (2026-11-03). Documentado en
[docs/05_experimentos.md — E19](../../docs/05_experimentos.md#e19--ablación-del-bloque-de-atrasos-en-xgboost)
(el análisis previo de redundancia y fuga está en E18).

**Resultado:** el bloque de atrasos es imprescindible (quitarlo cuesta 0.058 de
AUC), pero sus dos mitades son sustituibles entre sí. **Sin cambios en
`xgboost_v1`.** El script solo escribe en `resultados/`.

| Archivo | Contenido |
|---|---|
| `ablacion_bloque.py` | Escenarios A–D en CV 5×5 pareada con early stopping interno, y confirmación en validación (≈3 min con 16 núcleos) |
| `resultados/cv_por_particion.csv` | Métricas de las 100 ejecuciones |
| `resultados/resumen_cv.csv` | Media y desviación por escenario |
| `resultados/diferencias_pareadas.csv` | Diferencias contra A con IC 95 % de Nadeau–Bengio |
| `resultados/confirmacion_validacion.csv` | Una evaluación en validación (informativa) |
| `resultados/ablacion_bloque_atrasos.png` | Diferencias pareadas de AUC, KS y Log Loss |
| `resultados/config.json` | Hiperparámetros, escenarios y versiones |

Para reproducir, desde la raíz del proyecto:

```
python -u experiments/ablacion_bloque_xgb/ablacion_bloque.py
```
