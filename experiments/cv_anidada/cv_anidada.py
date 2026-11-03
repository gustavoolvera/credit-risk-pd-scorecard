# -*- coding: utf-8 -*-
"""
cv_anidada.py — ¿Esta inflado el AUC de CV de XGBoost por la seleccion?
Proyecto: Give Me Some Credit — Probability of Default

En 04_train_xgboost.py los hiperparametros se eligen con una rejilla evaluada
en un subconjunto de train (early stopping). Despues, la CV de 5 folds sobre
las 150,000 filas usa esos hiperparametros fijos: sus folds de prueba
contienen las filas que sirvieron para elegirlos.

Aqui se repite TODA la seleccion de 04 dentro de cada fold externo, usando
solo su parte de entrenamiento, y se compara fold a fold con la CV reportada:

  CV reportada : MEJORES y n_arboles de xgboost_v1, fijos en los 5 folds.
  CV anidada   : en cada fold, split 80/20 de su train -> rejilla de 11
                 configuraciones con early stopping -> la de mejor AUC en el
                 20% -> se reentrena con su train completo y ese n de arboles.

Mismos folds externos que 04 y 05: StratifiedKFold(5, shuffle, seed 42).
No toca ningun artefacto del pipeline: solo escribe en resultados/.

Se ejecuta con:  python -u experiments/cv_anidada/cv_anidada.py   (~5 min)
"""

import sys
import json
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src.evaluation import diferencia_pareada_cv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)
SEMILLA, T = 42, "SeriousDlqin2yrs"

# Rejilla identica a la de 04_train_xgboost.py (PASO 5)
REJILLA = [
    dict(max_depth=3, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=4, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=5, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=6, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=4, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, min_child_weight=20, subsample=0.8, colsample_bytree=0.6),
    dict(max_depth=7, min_child_weight=20, subsample=0.8, colsample_bytree=0.6),
    dict(max_depth=4, min_child_weight=10, subsample=0.9, colsample_bytree=0.9, reg_lambda=5),
    dict(max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5),
]

art = joblib.load(RAIZ / "artifacts" / "models" / "xgboost_v1.joblib")
COLS = list(art["columnas"])
MEJORES = {k: (float(v) if isinstance(v, np.floating) else v)
           for k, v in art["hiperparametros"].items()}

proc = RAIZ / "datasets" / "processed" / "trees"
datos = pd.concat([pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "val.parquet")])
X, y = datos[COLS], datos[T]
cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)


def seleccionar(X_tr, y_tr):
    """Replica la seleccion de 04 con los datos de un fold."""
    Xf, Xe, yf, ye = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr,
                                      random_state=SEMILLA)
    mejor = None
    for params in REJILLA:
        m = XGBClassifier(n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
                          eval_metric="auc", early_stopping_rounds=100, n_jobs=-1, **params)
        m.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
        if mejor is None or m.best_score > mejor[0]:
            mejor = (m.best_score, params, m.best_iteration + 1)
    return mejor


filas, t0 = [], time.time()
for k, (itr, ite) in enumerate(cv.split(X, y)):
    Xa, ya, Xb, yb = X.iloc[itr], y.iloc[itr], X.iloc[ite], y.iloc[ite]

    rep = XGBClassifier(n_estimators=art["n_arboles"], learning_rate=0.05,
                        random_state=SEMILLA, eval_metric="auc", n_jobs=-1, **MEJORES)
    auc_rep = roc_auc_score(yb, rep.fit(Xa, ya).predict_proba(Xb)[:, 1])

    auc_es, params, n_arb = seleccionar(Xa, ya)
    anid = XGBClassifier(n_estimators=n_arb, learning_rate=0.05, random_state=SEMILLA,
                         eval_metric="auc", n_jobs=-1, **params)
    auc_anid = roc_auc_score(yb, anid.fit(Xa, ya).predict_proba(Xb)[:, 1])

    filas.append({"fold": k + 1, "auc_reportada": auc_rep, "auc_anidada": auc_anid,
                  "params_elegidos": json.dumps(params), "n_arboles": n_arb,
                  "coincide_con_produccion": params == MEJORES})
    print(f"  fold {k + 1}: reportada {auc_rep:.5f} | anidada {auc_anid:.5f} | "
          f"{params} n={n_arb}  ({time.time() - t0:.0f}s)")

res = pd.DataFrame(filas)
res.to_csv(SALIDA / "auc_por_fold.csv", index=False)

n_tr = len(X) * 4 / 5
r = diferencia_pareada_cv(res.auc_anidada, res.auc_reportada, n_train=n_tr, n_test=len(X) / 5)
resumen = {
    "auc_reportada_media": round(res.auc_reportada.mean(), 5),
    "auc_reportada_de_ddof0": round(res.auc_reportada.std(ddof=0), 5),
    "auc_anidada_media": round(res.auc_anidada.mean(), 5),
    "auc_anidada_de_ddof0": round(res.auc_anidada.std(ddof=0), 5),
    "diferencia_anidada_menos_reportada": round(r["diferencia"], 5),
    "ic95_nadeau_bengio": [round(r["ic_inf"], 5), round(r["ic_sup"], 5)],
    "p": round(r["p"], 4),
    "folds_con_los_mismos_hiperparametros_que_produccion": int(res.coincide_con_produccion.sum()),
    "versiones": {"python": platform.python_version(), "xgboost": xgboost.__version__},
}
(SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
