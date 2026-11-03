# -*- coding: utf-8 -*-
"""
ablacion_bloque.py — ¿Cuanto aporta el bloque de atrasos a XGBoost?
Proyecto: Give Me Some Credit — Probability of Default

EXPERIMENTO AISLADO: solo escribe en experiments/ablacion_bloque_xgb/resultados/.
No toca xgboost_v1.joblib ni ningun artefacto del pipeline.

Bloque de atrasos (6 variables):
  originales : NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse,
               NumberOfTimes90DaysLate
  derivadas  : total_atrasos, flag_algun_atraso, peor_atraso

Escenarios
  A: todas (24)                       -> referencia = modelo de produccion
  B: sin el bloque (18)               -> aporte INCREMENTAL del bloque
  C: sin las 3 derivadas (21)         -> ¿son redundantes las derivadas?
  D: sin los 3 originales (21)        -> ¿bastan las derivadas?

Diseno
  - CV estratificada repetida 5x5 sobre las 120,000 de train; mismas
    particiones para los cuatro escenarios -> comparaciones pareadas.
  - Dentro de cada particion, 20% de su parte de ajuste se aparta para early
    stopping (igual que 04). La parte de prueba no decide nada.
  - Hiperparametros de produccion constantes (los eligio la rejilla de 04 CON
    todas las variables: si acaso, favorecen a A).
  - Incertidumbre: t-test con varianza corregida de Nadeau & Bengio (2003).
  - Validacion (30,000): una sola evaluacion de confirmacion, con el mismo
    split de early stopping que 04. No interviene en ninguna decision.

Se ejecuta con:  python experiments/ablacion_bloque_xgb/ablacion_bloque.py
"""

import sys
import json
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # signo "-" normal en los ejes
import sklearn
import xgboost
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss)

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src.evaluation import ks_statistic, error_calibracion, umbral_optimo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)
SEMILLA = 42
TARGET = "SeriousDlqin2yrs"
MARGEN, PERDIDA = 800, 10_000 * 0.60
UMBRAL = umbral_optimo(MARGEN, PERDIDA)

art = joblib.load(RAIZ / "artifacts" / "models" / "xgboost_v1.joblib")
COLS = list(art["columnas"])
PARAMS = dict(n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
              eval_metric="auc", early_stopping_rounds=100, n_jobs=-1,
              **{k: (float(v) if isinstance(v, np.floating) else v)
                 for k, v in art["hiperparametros"].items()})

ORIG = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"]
DERIV = ["total_atrasos", "flag_algun_atraso", "peor_atraso"]
assert set(ORIG + DERIV) <= set(COLS)
sin = lambda fuera: [c for c in COLS if c not in fuera]
ESC = {
    "A: todas": COLS,
    "B: sin bloque de atrasos": sin(ORIG + DERIV),
    "C: sin derivadas de atraso": sin(DERIV),
    "D: sin conteos originales": sin(ORIG),
}


def titulo(t):
    print("\n" + "=" * 79 + f"\n{t}\n" + "=" * 79)


def metricas(y, p):
    return {
        "AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
        "KS": ks_statistic(y, p)[0], "LogLoss": log_loss(y, p),
        "Brier": brier_score_loss(y, p), "Calib_pp": error_calibracion(y, p),
        "Benef_sol": (((y == 0) & (p < UMBRAL)).sum() * MARGEN
                      - ((y == 1) & (p < UMBRAL)).sum() * PERDIDA) / len(y),
    }


def entrenar(X, y, cols):
    """Early stopping con un 20% apartado de X (nunca con la parte de prueba)."""
    Xf, Xe, yf, ye = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
    m = XGBClassifier(**PARAMS)
    m.fit(Xf[cols], yf, eval_set=[(Xe[cols], ye)], verbose=False)
    return m


def nb(d, n_tr, n_te, nivel=0.95):
    d = np.asarray(d, dtype=float)
    se = np.sqrt(np.var(d, ddof=1) * (1 / len(d) + n_te / n_tr))
    q = stats.t.ppf(0.5 + nivel / 2, len(d) - 1)
    t = d.mean() / se if se > 0 else 0.0
    return d.mean(), d.mean() - q * se, d.mean() + q * se, 2 * stats.t.sf(abs(t), len(d) - 1)


# =============================================================================
# PASO 1 — DATOS
# =============================================================================
titulo("PASO 1 — DATOS (rama de arboles)")
tr = pd.read_parquet(RAIZ / "datasets" / "processed" / "trees" / "train.parquet")
va = pd.read_parquet(RAIZ / "datasets" / "processed" / "trees" / "val.parquet")
X_tr, y_tr = tr[COLS], tr[TARGET]
X_va, y_va = va[COLS], va[TARGET]
print(f"  Train {len(X_tr):,} | validacion {len(X_va):,} (solo confirmacion)")
print(f"  Hiperparametros de produccion: {art['hiperparametros']}")
for k, v in ESC.items():
    print(f"  {k:<30} {len(v)} variables")


# =============================================================================
# PASO 2 — CV 5x5 PAREADA
# =============================================================================
titulo("PASO 2 — CV ESTRATIFICADA 5x5 (mismas particiones para los 4 escenarios)")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEMILLA)
filas, t0 = [], time.time()
for k, (itr, ite) in enumerate(rskf.split(X_tr, y_tr)):
    Xa, ya = X_tr.iloc[itr], y_tr.iloc[itr]
    Xb, yb = X_tr.iloc[ite], y_tr.iloc[ite].to_numpy()
    for nombre, cols in ESC.items():
        m = entrenar(Xa, ya, cols)
        p = m.predict_proba(Xb[cols])[:, 1]
        filas.append({"fold": k, "escenario": nombre, "arboles": m.best_iteration + 1,
                      **metricas(yb, p)})
    print(f"  particion {k + 1:>2}/25  ({time.time() - t0:.0f}s)")
cv = pd.DataFrame(filas)
cv.to_csv(SALIDA / "cv_por_particion.csv", index=False)
n_te = len(X_tr) // 5
n_tr = len(X_tr) - n_te


# =============================================================================
# PASO 3 — RESUMEN Y DIFERENCIAS PAREADAS CONTRA A
# =============================================================================
titulo("PASO 3 — RESUMEN Y DIFERENCIAS PAREADAS (Nadeau-Bengio, IC 95%)")
METR = ["AUC", "KS", "LogLoss", "AP", "Brier", "Calib_pp", "Benef_sol", "arboles"]
resumen = cv.groupby("escenario", sort=False)[METR].agg(["mean", "std"])
resumen.columns = [f"{a}_{b}" for a, b in resumen.columns]
resumen.to_csv(SALIDA / "resumen_cv.csv")
print(resumen[[f"{m}_mean" for m in METR]].round(5).to_string())

base = cv[cv.escenario == "A: todas"].set_index("fold")
par = []
for nombre in list(ESC)[1:]:
    o = cv[cv.escenario == nombre].set_index("fold")
    for mtr in METR[:-1]:
        d = (o[mtr] - base[mtr]).to_numpy()
        media, lo, hi, p = nb(d, n_tr, n_te)
        peor = (d > 0) if mtr in ("LogLoss", "Brier", "Calib_pp") else (d < 0)
        par.append({"escenario": nombre, "metrica": mtr, "dif": media, "IC95_lo": lo,
                    "IC95_hi": hi, "p": p, "pct_particiones_peor": peor.mean() * 100,
                    "dif_relativa_pct": media / base[mtr].mean() * 100})
par = pd.DataFrame(par)
par.to_csv(SALIDA / "diferencias_pareadas.csv", index=False)
for nombre, g in par.groupby("escenario", sort=False):
    print(f"\n  {nombre}  (vs A)")
    print(g.drop(columns="escenario").to_string(index=False, float_format=lambda v: f"{v:.5f}"))


# =============================================================================
# PASO 4 — CONFIRMACION EN VALIDACION (no decide nada)
# =============================================================================
titulo("PASO 4 — CONFIRMACION EN VALIDACION (mismo split de early stopping que 04)")
conf, pA = [], None
for nombre, cols in ESC.items():
    m = entrenar(X_tr, y_tr, cols)
    p = m.predict_proba(X_va[cols])[:, 1]
    if pA is None:
        pA = p
    yv = y_va.to_numpy()
    conf.append({"escenario": nombre, "arboles": m.best_iteration + 1, **metricas(yv, p),
                 "dec_cambiadas_vs_A": int(((p < UMBRAL) != (pA < UMBRAL)).sum())})
conf = pd.DataFrame(conf)
conf["Benef_total"] = conf["Benef_sol"] * len(y_va)
conf.to_csv(SALIDA / "confirmacion_validacion.csv", index=False)
print(conf.round(5).to_string(index=False))
prod = roc_auc_score(y_va, art["modelo"].predict_proba(X_va)[:, 1])
print(f"\n  Control: AUC del modelo de produccion en validacion = {prod:.5f} "
      f"(A reentrenado = {conf.loc[0, 'AUC']:.5f})")


# =============================================================================
# PASO 5 — GRAFICA
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
orden = list(ESC)[1:][::-1]
for ax, (mtr, nom, esc) in zip(axes, [("AUC", "Δ AUC", 1), ("KS", "Δ KS", 1),
                                      ("LogLoss", "Δ Log Loss (×1e3)", 1e3)]):
    g = par[par.metrica == mtr].set_index("escenario").loc[orden]
    m_, lo, hi = g.dif * esc, g.IC95_lo * esc, g.IC95_hi * esc
    malo = (lo > 0) if mtr == "LogLoss" else (hi < 0)
    for i in range(len(g)):
        col = "#a8392f" if malo.iloc[i] else "#7a3a6a"
        ax.errorbar(m_.iloc[i], i, xerr=[[m_.iloc[i] - lo.iloc[i]], [hi.iloc[i] - m_.iloc[i]]],
                    fmt="o", color=col, ecolor=col, capsize=3)
        ax.annotate(f"{m_.iloc[i]:+.4f}" if esc == 1 else f"{m_.iloc[i]:+.2f}",
                    (m_.iloc[i], i), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(nom, fontsize=10)
    ax.grid(alpha=0.3, axis="x")
axes[0].set_yticks(range(len(orden)))
axes[0].set_yticklabels(orden, fontsize=9)
fig.suptitle("XGBoost — ablación del bloque de atrasos vs A (CV 5×5, IC 95% Nadeau–Bengio; "
             "rojo = deterioro significativo)", fontsize=10.5)
fig.text(0.01, 0.01, "Log Loss: positivo = peor. AUC y KS: negativo = peor.",
         fontsize=8, color="dimgray")
plt.tight_layout(rect=(0, 0.05, 1, 0.92))
plt.savefig(SALIDA / "ablacion_bloque_atrasos.png", dpi=130)
plt.close()

(SALIDA / "config.json").write_text(json.dumps({
    "cv": "RepeatedStratifiedKFold(5x5, random_state=42) sobre train; ES con 20% del ajuste",
    "params_xgb": PARAMS, "umbral": UMBRAL,
    "escenarios": ESC,
    "versiones": {"python": platform.python_version(), "xgboost": xgboost.__version__,
                  "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                  "pandas": pd.__version__},
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Resultados en {SALIDA.relative_to(RAIZ)}  ({time.time() - t0:.0f}s)")
