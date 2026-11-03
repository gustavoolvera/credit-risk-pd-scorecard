# -*- coding: utf-8 -*-
"""
seleccion.py — Eleccion del modelo final con CV dentro de train
Proyecto: Give Me Some Credit — Probability of Default

Regla (docs/04_modelado.md §4.7.2): entre los modelos con AUC medio de CV a
menos de MARGEN_AUC del mejor, se elige el de menor error de calibracion medio.
Las metricas salen de una CV estratificada repetida sobre las filas de train,
asi que el conjunto de validacion no interviene en la eleccion.

La regla se fijo antes de conocer su resultado. Con modelos estadisticamente
indistinguibles, el desempate es arbitrario; lo que la regla garantiza es que
no se elige mirando validacion ni a posteriori.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from . import woe as woe_mod
from .evaluation import error_calibracion

MARGEN_AUC = 0.002


def regla_seleccion(medias, margen=MARGEN_AUC):
    """
    medias: DataFrame indexado por modelo con columnas 'AUC' y 'Calib_pp'.
    Devuelve (elegido, lista de elegibles por AUC).
    """
    if not {"AUC", "Calib_pp"} <= set(medias.columns):
        raise ValueError("medias necesita las columnas 'AUC' y 'Calib_pp'")
    elegibles = medias[medias["AUC"] >= medias["AUC"].max() - margen]
    return elegibles["Calib_pp"].idxmin(), list(elegibles.index)


def estimadores(arts, semilla=42):
    """Estimadores sin entrenar, con los hiperparametros de produccion."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    a_log, a_xgb, a_lgb = arts["logistic_v1"], arts["xgboost_v1"], arts["lightgbm_v1"]
    return {
        "logistic_v1": LogisticRegression(**a_log["modelo"].get_params()),
        "xgboost_v1": XGBClassifier(n_estimators=a_xgb["n_arboles"], learning_rate=0.05,
                                    random_state=semilla, eval_metric="auc", n_jobs=-1,
                                    **a_xgb["hiperparametros"]),
        "lightgbm_v1": LGBMClassifier(n_estimators=a_lgb["n_arboles"], learning_rate=0.05,
                                      random_state=semilla, n_jobs=-1, verbose=-1,
                                      bagging_freq=1, **a_lgb["hiperparametros"]),
    }


def cv_en_train(X_crudo, X_trees, y, arts, umbral, margen_neg, perdida,
                n_splits=5, n_repeats=5, semilla=42):
    """
    CV pareada de los tres modelos sobre train.

    X_crudo: train limpio (antes de WoE); la LogReg re-aprende las tablas WoE
             dentro de cada particion, para no evaluar con WoE visto en el fold.
    X_trees: train de la rama de arboles (sin estado aprendido).
    Devuelve una fila por (particion, modelo) con AUC, Calib_pp, Brier,
    LogLoss y beneficio por solicitud al umbral dado.
    """
    if not X_crudo.index.equals(X_trees.index):
        raise ValueError("X_crudo y X_trees deben tener las mismas filas en el mismo orden")
    cols_woe = arts["logistic_v1"]["columnas"]
    orig = [c[4:] for c in cols_woe]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    filas = []
    for k, (itr, ite) in enumerate(rskf.split(X_crudo, y)):
        ya, yb = y.iloc[itr], y.iloc[ite].to_numpy()
        tablas = woe_mod.aprender_woe(X_crudo.iloc[itr][orig], ya)
        Wa = woe_mod.aplicar_woe(X_crudo.iloc[itr][orig], tablas)[cols_woe]
        Wb = woe_mod.aplicar_woe(X_crudo.iloc[ite][orig], tablas)[cols_woe]
        for nombre, m in estimadores(arts, semilla).items():
            if nombre == "logistic_v1":
                Xa, Xb = Wa, Wb
            else:
                cols = arts[nombre]["columnas"]
                Xa, Xb = X_trees.iloc[itr][cols], X_trees.iloc[ite][cols]
            p = m.fit(Xa, ya).predict_proba(Xb)[:, 1]
            aprob = p < umbral
            filas.append({
                "fold": k, "modelo": nombre,
                "AUC": roc_auc_score(yb, p), "Calib_pp": error_calibracion(yb, p),
                "Brier": brier_score_loss(yb, p), "LogLoss": log_loss(yb, p),
                "Benef_sol": (((yb == 0) & aprob).sum() * margen_neg
                              - ((yb == 1) & aprob).sum() * perdida) / len(yb),
            })
    return pd.DataFrame(filas)
