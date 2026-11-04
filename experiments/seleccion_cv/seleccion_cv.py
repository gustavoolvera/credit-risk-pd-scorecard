# -*- coding: utf-8 -*-
"""
seleccion_cv.py: ¿Que modelo elegiria una regla que no use validacion?
Proyecto: Give Me Some Credit: Probability of Default

REGLA FIJADA ANTES DE EJECUTAR (2026-11-03)
  Es la regla vigente de 06_evaluation.py con UN solo cambio: las metricas
  salen de una CV dentro de train, no del conjunto de validacion.

    Regla vigente (06):  entre los modelos con AUC_val >= max(AUC_val) - 0.002,
                         el de menor error de calibracion en validacion.
    Regla CV (esta):     entre los modelos con AUC_cv  >= max(AUC_cv)  - 0.002,
                         el de menor error de calibracion medio en la CV.

  Mismo margen (0.002), mismo desempate, misma metrica de calibracion
  (error_calibracion, deciles). No se ajusta nada tras ver los resultados.

DISENO
  - Solo las 120,000 filas de train. Validacion (30,000) no interviene.
  - CV estratificada repetida 5x5, mismas particiones para los 3 modelos.
  - LogReg: tablas WoE re-aprendidas dentro de cada particion (cortes fijos).
  - Boosters: hiperparametros y numero de arboles de produccion (elegidos en
    04/05 con subconjuntos de train).
  - Diferencias pareadas con IC 95% de Nadeau-Bengio.

No modifica el pipeline ni el modelo elegido: solo escribe en resultados/.

Se ejecuta con:  python -u experiments/seleccion_cv/seleccion_cv.py   (~3 min)
"""

import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src import preprocessing as pp
from src import woe as woe_mod
from src.artifacts import ruta_datos
from src.evaluation import (error_calibracion, umbral_optimo, diferencia_pareada_cv)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)
SEMILLA, T = 42, pp.TARGET
MARGEN_AUC = 0.002                       # el mismo de 06_evaluation.py
M, L = 800, 6000
UMBRAL = umbral_optimo(M, L)
MOD = RAIZ / "artifacts" / "models"
NOMBRES = {"logistic_v1": "LogReg (WoE)", "xgboost_v1": "XGBoost", "lightgbm_v1": "LightGBM"}

# --- Datos: mismo split que 02; solo train ------------------------------------
raw = pd.read_csv(ruta_datos(RAIZ) / "cs-training.csv", index_col=0)
raw.index.name = "row_id"
X_raw, _, y, _ = train_test_split(raw.drop(columns=[T]), raw[T], test_size=0.20,
                                  stratify=raw[T], random_state=SEMILLA)
X_crudo = pp.crear_features(pp.limpiar(X_raw))
trees = pd.read_parquet(RAIZ / "datasets" / "processed" / "trees" / "train.parquet")
assert trees.index.equals(X_crudo.index)

a_log, a_xgb, a_lgb = (joblib.load(MOD / f"{n}.joblib") for n in NOMBRES)
COLS_WOE = a_log["columnas"]
ORIG_WOE = [c[4:] for c in COLS_WOE]


def modelos():
    return {
        "logistic_v1": LogisticRegression(**a_log["modelo"].get_params()),
        "xgboost_v1": XGBClassifier(n_estimators=a_xgb["n_arboles"], learning_rate=0.05,
                                    random_state=SEMILLA, eval_metric="auc", n_jobs=-1,
                                    **a_xgb["hiperparametros"]),
        "lightgbm_v1": LGBMClassifier(n_estimators=a_lgb["n_arboles"], learning_rate=0.05,
                                      random_state=SEMILLA, n_jobs=-1, verbose=-1,
                                      bagging_freq=1, **a_lgb["hiperparametros"]),
    }


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEMILLA)
filas, t0 = [], time.time()
for k, (itr, ite) in enumerate(rskf.split(X_crudo, y)):
    ya, yb = y.iloc[itr], y.iloc[ite].to_numpy()
    tablas = woe_mod.aprender_woe(X_crudo.iloc[itr][ORIG_WOE], ya)
    Wa = woe_mod.aplicar_woe(X_crudo.iloc[itr][ORIG_WOE], tablas)[COLS_WOE]
    Wb = woe_mod.aplicar_woe(X_crudo.iloc[ite][ORIG_WOE], tablas)[COLS_WOE]
    for nombre, m in modelos().items():
        if nombre == "logistic_v1":
            Xa, Xb = Wa, Wb
        else:
            cols = (a_xgb if nombre == "xgboost_v1" else a_lgb)["columnas"]
            Xa, Xb = trees.iloc[itr][cols], trees.iloc[ite][cols]
        p = m.fit(Xa, ya).predict_proba(Xb)[:, 1]
        filas.append({"fold": k, "modelo": NOMBRES[nombre],
                      "AUC": roc_auc_score(yb, p), "Calib_pp": error_calibracion(yb, p),
                      "Brier": brier_score_loss(yb, p), "LogLoss": log_loss(yb, p),
                      "Benef_sol": (((yb == 0) & (p < UMBRAL)).sum() * M
                                    - ((yb == 1) & (p < UMBRAL)).sum() * L) / len(yb)})
    print(f"  particion {k + 1:>2}/25 ({time.time() - t0:.0f}s)")

cv = pd.DataFrame(filas)
cv.to_csv(SALIDA / "cv_por_particion.csv", index=False)
res = cv.groupby("modelo")[["AUC", "Calib_pp", "Brier", "LogLoss", "Benef_sol"]].agg(["mean", "std"])
res.columns = [f"{a}_{b}" for a, b in res.columns]
res.to_csv(SALIDA / "resumen_cv.csv")
print("\n" + res.round(5).to_string())

# --- Las dos reglas ---------------------------------------------------------
media = cv.groupby("modelo")[["AUC", "Calib_pp"]].mean()
emp_cv = media[media.AUC >= media.AUC.max() - MARGEN_AUC]
eleg_cv = emp_cv.Calib_pp.idxmin()

# La regla anterior, aplicada a las metricas de validacion que publica 06
comp = pd.read_csv(RAIZ / "artifacts" / "metrics" / "comparacion_final.csv")
comp = comp[comp.modelo != "Ensemble (promedio)"].set_index("modelo")
emp_val = comp[comp.AUC_val >= comp.AUC_val.max() - MARGEN_AUC]
eleg_val = emp_val.calib_pp.idxmin()
print(f"\n  Regla con validacion (anterior): {eleg_val}")
print(f"  Regla CV (solo train)     : {eleg_cv}   (dentro del margen: {list(emp_cv.index)})")

# --- Evidencia: diferencias pareadas -----------------------------------------
n_te = len(y) / 5
pares = []
for a, b in [("XGBoost", "LightGBM"), ("XGBoost", "LogReg (WoE)"), ("LightGBM", "LogReg (WoE)")]:
    A = cv[cv.modelo == a].set_index("fold"); B = cv[cv.modelo == b].set_index("fold")
    for mtr in ["AUC", "Calib_pp", "Brier", "Benef_sol"]:
        r = diferencia_pareada_cv(A[mtr], B[mtr], n_train=len(y) - n_te, n_test=n_te)
        pares.append({"comparacion": f"{a} - {b}", "metrica": mtr, **r})
pares = pd.DataFrame(pares)
pares.to_csv(SALIDA / "diferencias_pareadas.csv", index=False)
print("\n" + pares.drop(columns="gl").to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

(SALIDA / "eleccion.json").write_text(json.dumps({
    "regla_cv": f"AUC_cv >= max - {MARGEN_AUC}; desempate por menor Calib_pp medio (CV 5x5 en train)",
    "dentro_del_margen": list(emp_cv.index),
    "elegido_regla_cv": eleg_cv,
    "elegido_regla_validacion": eleg_val,
}, indent=2, ensure_ascii=False), encoding="utf-8")
