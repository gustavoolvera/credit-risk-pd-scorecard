# Validación cruzada anidada (E11)

Documentado en
[docs/05_experimentos.md — E11](../../docs/05_experimentos.md#e11--validación-cruzada-anidada).

**Pregunta.** La CV de XGBoost (0.86628) usa hiperparámetros elegidos con filas
que después aparecen en sus folds de prueba. ¿Infla eso el AUC?

**Método.** Mismos 5 folds externos que `04` y `05`. En cada fold se repite la
selección completa de `04` (rejilla de 11 configuraciones con early stopping)
usando solo su parte de entrenamiento, y se evalúa en su parte de prueba.

**Resultado.** CV anidada 0.86647 frente a 0.86628 reportada: diferencia
+0.00018, IC 95 % (Nadeau–Bengio) [-0.00084, +0.00120]. No se detecta
optimismo. Solo 1 de 5 folds elige la misma configuración que producción: el
AUC es casi plano en la rejilla.

| Archivo | Contenido |
|---|---|
| `cv_anidada.py` | El experimento (~3 min con 16 núcleos). No modifica el pipeline |
| `resultados/auc_por_fold.csv` | AUC reportado y anidado por fold, configuración elegida |
| `resultados/resumen.json` | Medias, diferencia, IC, p y versiones |

```
python -u experiments/cv_anidada/cv_anidada.py
```

Requiere haber ejecutado antes `02` y `04` (lee `datasets/processed/trees/` y
`artifacts/models/xgboost_v1.joblib`).
