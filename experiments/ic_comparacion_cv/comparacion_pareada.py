# -*- coding: utf-8 -*-
"""
comparacion_pareada.py — AUC por fold de la CV de 05 y diferencias pareadas
Proyecto: Give Me Some Credit — Probability of Default

05_train_lightgbm.py guarda solo la media y la desviacion de cada modelo, no
los AUC de cada fold. Este script los reconstruye con el MISMO procedimiento
(mismos datos, mismo StratifiedKFold, mismos estimadores) y comprueba que
reproduce exactamente las cifras guardadas antes de usarlos.

Los modelos de cada fold existen solo en memoria: no se toca ningun modelo de
produccion ni ningun archivo de artifacts/. Solo escribe en resultados/.

Se ejecuta con:  python experiments/ic_comparacion_cv/comparacion_pareada.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src.evaluation import diferencia_pareada_cv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)
T, SEMILLA = "SeriousDlqin2yrs", 42
PROC, MOD = RAIZ / "datasets" / "processed", RAIZ / "artifacts" / "models"

# --- Igual que 05_train_lightgbm.py, PASO 7 ----------------------------------
trees = pd.concat([pd.read_parquet(PROC / "trees" / f"{s}.parquet") for s in ("train", "val")])
woe = pd.concat([pd.read_parquet(PROC / "woe" / f"{s}.parquet") for s in ("train", "val")])
assert trees.index.equals(woe.index)
cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
a_log, a_xgb, a_lgb = (joblib.load(MOD / f"{n}.joblib")
                       for n in ("logistic_v1", "xgboost_v1", "lightgbm_v1"))

auc = {
    "LogReg (WoE)": cross_val_score(
        LogisticRegression(max_iter=5000, random_state=SEMILLA),
        woe[a_log["columnas"]], woe[T], cv=cv, scoring="roc_auc"),
    "XGBoost": cross_val_score(
        XGBClassifier(n_estimators=a_xgb["n_arboles"], learning_rate=0.05,
                      random_state=SEMILLA, eval_metric="auc", n_jobs=-1,
                      **a_xgb["hiperparametros"]),
        trees[a_xgb["columnas"]], trees[T], cv=cv, scoring="roc_auc"),
    "LightGBM": cross_val_score(
        LGBMClassifier(n_estimators=a_lgb["n_arboles"], learning_rate=0.05,
                       random_state=SEMILLA, n_jobs=-1, verbose=-1, bagging_freq=1,
                       **a_lgb["hiperparametros"]),
        trees[a_lgb["columnas"]], trees[T], cv=cv, scoring="roc_auc"),
}

# --- 1. ¿Reproduce lo guardado? ---------------------------------------------
guardado = pd.read_csv(RAIZ / "artifacts" / "metrics" / "comparacion_modelos_cv.csv").set_index("modelo")
print("REPRODUCCION DE comparacion_modelos_cv.csv (media y DE con ddof=0)")
for n, s in auc.items():
    ok = (round(s.mean(), 5) == guardado.loc[n, "auc_cv_media"]
          and round(s.std(), 5) == guardado.loc[n, "auc_cv_desv"])
    print(f"  {n:<14} media {s.mean():.5f} DE0 {s.std():.5f}  "
          f"guardado {guardado.loc[n, 'auc_cv_media']:.5f} {guardado.loc[n, 'auc_cv_desv']:.5f}  "
          f"-> {'OK' if ok else 'NO COINCIDE'}")
    assert ok, f"{n}: los folds reconstruidos no reproducen las cifras guardadas"

folds = pd.DataFrame(auc)
folds.index = [f"fold_{i + 1}" for i in range(len(folds))]
folds.to_csv(SALIDA / "auc_por_fold.csv")
print("\nAUC por fold:\n" + folds.round(5).to_string())

# --- 2. ¿La correccion corresponde al diseno? -------------------------------
tam = [(len(tr), len(te)) for tr, te in cv.split(trees, trees[T])]
n_tr, n_te = tam[0]
assert all(t == tam[0] for t in tam), "los folds no tienen el mismo tamano"
k = cv.get_n_splits()
print(f"\nDISENO: {k}-fold estratificado, 1 repeticion, J = {k} diferencias pareadas")
print(f"  n_train = {n_tr:,}  n_test = {n_te:,}  n_test/n_train = {n_te / n_tr:.4f}"
      f"  (= 1/(k-1) = {1 / (k - 1):.4f})")
print(f"  var corregida = (1/J + n_test/n_train) * s^2 = ({1 / k:.2f} + {n_te / n_tr:.2f}) * s^2"
      f"  | t con {k - 1} g.l.")

# --- 3. Diferencias pareadas -------------------------------------------------
filas = []
for a, b in [("XGBoost", "LogReg (WoE)"), ("LightGBM", "LogReg (WoE)"), ("XGBoost", "LightGBM")]:
    r = diferencia_pareada_cv(auc[a], auc[b], n_train=n_tr, n_test=n_te)
    ing = diferencia_pareada_cv(auc[a], auc[b], n_train=1e15, n_test=1)   # sin corregir
    filas.append({"comparacion": f"{a} - {b}", **r,
                  "ic_inf_sin_corregir": ing["ic_inf"], "ic_sup_sin_corregir": ing["ic_sup"],
                  "p_sin_corregir": ing["p"]})
tabla = pd.DataFrame(filas)
tabla.to_csv(SALIDA / "diferencias_pareadas.csv", index=False)
print("\nDIFERENCIAS PAREADAS (IC 95%)")
print(tabla.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
