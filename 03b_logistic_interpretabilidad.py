# -*- coding: utf-8 -*-
"""
03b_logistic_interpretabilidad.py — ¿Que variables pesan en el baseline?
Proyecto: Give Me Some Credit — Probability of Default

Complementa a 03_train_logistic.py con dos lecturas del modelo YA entrenado:

  1. Coeficientes estandarizados  beta * sigma(WoE)
     -> tamano del efecto sobre el log-odds por +1 desviacion estandar.
  2. Permutation Importance (ROC-AUC y Average Precision)
     -> cuanto empeora la discriminacion si se rompe la relacion entre
        una variable y el target.

No reentrena nada ni reajusta el WoE: solo lee logistic_v1.joblib y los
parquet WoE, y llama a predict_proba. Por eso se puede regenerar en segundos
sin repetir la seleccion de variables de 03.

Se ejecuta con:  python 03b_logistic_interpretabilidad.py
"""

import sys
import hashlib
import json
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

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation import coeficientes_estandarizados

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed" / "woe"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
for d in (METRICAS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
SEMILLA = 42
N_REPEATS = 20

AZUL, ROJO, GRIS = "#3a4f7a", "#a8392f", "#9a9a9a"


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


def check(cond, msg):
    print(f"  [{'OK' if cond else 'FALLO'}] {msg}")
    if not cond:
        raise AssertionError(msg)


def md5(ruta):
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def legible(col):
    return col.replace("woe_", "")


# =============================================================================
# PASO 1 — CARGA Y COMPROBACIONES
# =============================================================================
titulo("PASO 1 — MODELO, DATOS Y COMPROBACIONES")

RUTA_MODELO = MODELOS / "logistic_v1.joblib"
RUTA_METRICAS = METRICAS / "logistic_v1.json"
huella_antes = {p.name: md5(p) for p in (RUTA_MODELO, RUTA_METRICAS)}

art = joblib.load(RUTA_MODELO)
modelo, COLS = art["modelo"], list(art["columnas"])

# train: con el que se ajustaron el WoE y el modelo -> de aqui sale sigma.
# val:   nunca participo en ningun ajuste -> aqui se mide la importancia.
# test:  cs-test.csv de Kaggle, sin target -> no sirve (ni se toca).
train = pd.read_parquet(PROC / "train.parquet")
val = pd.read_parquet(PROC / "val.parquet")
X_tr, y_tr = train[COLS], train[TARGET]
X_val, y_val = val[COLS], val[TARGET]

print(f"  Modelo        : {type(modelo).__name__} ({art['subconjunto']})")
print(f"  Entrenamiento : {X_tr.shape[0]:>7,} filas x {X_tr.shape[1]} variables")
print(f"  Validacion    : {X_val.shape[0]:>7,} filas x {X_val.shape[1]} variables\n")

check(list(modelo.feature_names_in_) == COLS,
      "columnas guardadas = orden de entrenamiento del modelo")
check(modelo.coef_.shape == (1, len(COLS)), f"un coeficiente por variable ({len(COLS)})")
check(list(modelo.classes_) == [0, 1], "classes_ = [0, 1] -> predict_proba[:, 1] = P(default)")
check(X_tr.isna().sum().sum() == 0 and X_val.isna().sum().sum() == 0,
      "sin NaN en train ni validacion")
check(np.isfinite(X_val.to_numpy()).all(), "sin infinitos en validacion")
check(len(set(train.index) & set(val.index)) == 0, "train y validacion no comparten personas")

# La PD que reporta 03 es predict_proba[:, 1]; los scorers usan exactamente eso.
p_val = modelo.predict_proba(X_val)[:, 1]
AUC_BASE = roc_auc_score(y_val, p_val)
AP_BASE = average_precision_score(y_val, p_val)
guardadas = json.loads(RUTA_METRICAS.read_text())
check(abs(AUC_BASE - guardadas["auc_val"]) < 1e-5,
      f"AUC base {AUC_BASE:.5f} = logistic_v1.json ({guardadas['auc_val']})")
check(abs(AP_BASE - guardadas["average_precision_val"]) < 1e-5,
      f"AP base  {AP_BASE:.5f} = logistic_v1.json ({guardadas['average_precision_val']})")


# =============================================================================
# PASO 2 — COEFICIENTES ESTANDARIZADOS
# =============================================================================
titulo("PASO 2 — COEFICIENTES ESTANDARIZADOS  beta * sigma(WoE)")

# Todas las variables estan en log-odds (WoE), pero con dispersiones muy
# distintas. sigma se toma de TRAIN: es la escala con la que el modelo
# aprendio sus betas.
coef = coeficientes_estandarizados(modelo, X_tr, COLS)
coef["rank_beta"] = coef["beta"].abs().rank(ascending=False).astype(int)
coef["rank_beta_std"] = coef["beta_std"].abs().rank(ascending=False).astype(int)

print(coef.sort_values("rank_beta_std").assign(variable=lambda d: d.variable.map(legible))
      .round(4).to_string(index=False))


# =============================================================================
# PASO 3 — PERMUTATION IMPORTANCE EN VALIDACION
# =============================================================================
titulo("PASO 3 — PERMUTATION IMPORTANCE (validacion)")

# Permutar la columna WoE equivale EXACTAMENTE a permutar la variable original
# y transformarla despues: el WoE es un mapeo bin -> valor columna a columna.
# Los bins y las tablas WoE no se tocan.
scorers = {
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "average_precision": make_scorer(average_precision_score,
                                     response_method="predict_proba"),
}
pi = permutation_importance(modelo, X_val, y_val, scoring=scorers,
                            n_repeats=N_REPEATS, random_state=SEMILLA, n_jobs=1)

perm = pd.DataFrame({
    "variable": COLS,
    "auc_drop_mean": pi["roc_auc"].importances_mean,
    "auc_drop_std": pi["roc_auc"].importances_std,
    "ap_drop_mean": pi["average_precision"].importances_mean,
    "ap_drop_std": pi["average_precision"].importances_std,
})
perm["rank_auc"] = perm["auc_drop_mean"].rank(ascending=False).astype(int)
perm["rank_ap"] = perm["ap_drop_mean"].rank(ascending=False).astype(int)

print(f"  n_repeats = {N_REPEATS} | random_state = {SEMILLA} | "
      f"AUC base = {AUC_BASE:.5f} | AP base = {AP_BASE:.5f}\n")
print(perm.sort_values("rank_auc").assign(variable=lambda d: d.variable.map(legible))
      .round(5).to_string(index=False))

negativas = perm.loc[perm["auc_drop_mean"] < 0, "variable"].map(legible).tolist()
print(f"\n  Caida de AUC negativa (permutar no empeora): {negativas or 'ninguna'}")


# --- Correlaciones: lo que la permutacion individual no ve ------------------
corr = X_tr.corr()
np.fill_diagonal(corr.values, np.nan)
for c in perm.sort_values("rank_auc")["variable"]:
    socio = corr[c].abs().idxmax()
    print(f"    {legible(c):<42} rho = {corr.loc[c, socio]:+.2f}  con {legible(socio)}")


# =============================================================================
# PASO 4 — TABLA COMPARATIVA
# =============================================================================
titulo("PASO 4 — TABLA COMPARATIVA")

tabla = coef.merge(perm, on="variable").sort_values("rank_auc").reset_index(drop=True)
tabla.insert(1, "nombre", tabla["variable"].map(legible))
tabla["dif_rank"] = tabla["rank_beta_std"] - tabla["rank_auc"]

print(tabla[["nombre", "beta", "beta_std", "rank_beta", "rank_beta_std",
             "auc_drop_mean", "rank_auc", "ap_drop_mean", "rank_ap"]]
      .round(4).to_string(index=False))

rho_s = tabla[["rank_beta_std", "rank_auc"]].corr(method="spearman").iloc[0, 1]
print(f"\n  Concordancia de rankings beta_std vs caida AUC (Spearman): {rho_s:.3f}")

tabla.to_csv(METRICAS / "logistic_interpretabilidad.csv", index=False)

resumen = {
    "modelo": "logistic_v1.joblib",
    "subconjunto": art["subconjunto"],
    "conjunto_sigma": f"train WoE ({len(X_tr)} filas)",
    "conjunto_permutacion": f"validacion WoE ({len(X_val)} filas)",
    "auc_base_val": round(float(AUC_BASE), 5),
    "ap_base_val": round(float(AP_BASE), 5),
    "n_repeats": N_REPEATS,
    "random_state": SEMILLA,
    "spearman_rank_beta_std_vs_auc": round(float(rho_s), 4),
    "versiones": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    },
}
(METRICAS / "logistic_interpretabilidad.json").write_text(
    json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# PASO 5 — GRAFICAS
# =============================================================================
titulo("PASO 5 — GRAFICAS")

# --- Grafica 1: coeficientes estandarizados ---------------------------------
c = coef.sort_values("beta_std", key=abs, ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
colores = [ROJO if v > 0 else AZUL for v in c["beta_std"]]
ax.barh(range(len(c)), c["beta_std"], color=colores, height=0.65)
ax.set_yticks(range(len(c)))
ax.set_yticklabels(c["variable"].map(legible), fontsize=8)
ax.axvline(0, color="black", lw=0.8)

lim = c["beta_std"].abs().max() * 1.25
ax.set_xlim(-lim if (c["beta_std"] < 0).any() else -lim * 0.1,
            lim if (c["beta_std"] > 0).any() else lim * 0.1)
for i, v in enumerate(c["beta_std"]):
    ax.text(v + (0.01 * lim if v >= 0 else -0.01 * lim), i, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)

ax.set_xlabel(r"$\beta_j \cdot \sigma(\mathrm{WoE}_j)$   (cambio en log-odds por +1 desviación estándar)")
ax.set_title("Coeficientes estandarizados: Regresión Logística WoE")
ax.grid(alpha=0.3, axis="x")
fig.text(0.01, 0.01,
         "σ calculada en train. Negativo = más WoE (bin más seguro) reduce la PD.  "
         "Mide tamaño del efecto en el predictor lineal, no importancia predictiva.",
         fontsize=7.5, color="dimgray")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(PLOTS / "03b_logistic_coef_estandarizados.png", dpi=130)
plt.close()

# --- Grafica 2: permutation importance --------------------------------------
p = perm.sort_values("auc_drop_mean", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10.5, 6))
neg = p["auc_drop_mean"] < 0
ax.barh(range(len(p)), p["auc_drop_mean"], xerr=p["auc_drop_std"],
        color=[GRIS if n else AZUL for n in neg], height=0.65,
        hatch=["//" if n else "" for n in neg], edgecolor="white",
        error_kw=dict(ecolor="black", lw=0.8, capsize=3))
ax.set_yticks(range(len(p)))
ax.set_yticklabels(p["variable"].map(legible), fontsize=8)
ax.axvline(0, color="black", lw=0.8)

top = (p["auc_drop_mean"] + p["auc_drop_std"]).max()
ax.set_xlim(min(0, p["auc_drop_mean"].min()) - 0.05 * top, top * 1.22)
for i, r in p.iterrows():
    x = max(r.auc_drop_mean + r.auc_drop_std, 0) + 0.01 * top
    if r.auc_drop_mean < 0:
        ax.text(x, i, f"{r.auc_drop_mean:.4f}  (negativa)", va="center",
                fontsize=8, color="dimgray", style="italic")
    else:
        ax.text(x, i, f"{r.auc_drop_mean:.4f}", va="center", fontsize=8)
    # Average Precision como columna complementaria a la derecha
    ax.annotate(f"{r.ap_drop_mean:+.4f}", xy=(1.02, i), xycoords=("axes fraction", "data"),
                va="center", fontsize=8, color=ROJO)
ax.annotate("Δ AP", xy=(1.02, len(p) - 0.3), xycoords=("axes fraction", "data"),
            fontsize=8.5, color=ROJO, fontweight="bold")

ax.set_xlabel("Caída media de ROC-AUC al permutar la variable  (± 1 desv. estándar, "
              f"{N_REPEATS} repeticiones)")
ax.set_title("Permutation Feature Importance: Regresión Logística WoE (validación)")
ax.grid(alpha=0.3, axis="x")
nota = (f"AUC base = {AUC_BASE:.4f} | AP base = {AP_BASE:.4f} | n = {len(X_val):,}.  "
        "Gris/(negativa) = permutar no empeora el modelo.  "
        "Variables correlacionadas reparten su importancia.")
fig.text(0.01, 0.01, nota, fontsize=7.5, color="dimgray")
plt.tight_layout(rect=(0, 0.03, 0.93, 1))
plt.savefig(PLOTS / "03b_logistic_permutation_importance.png", dpi=130)
plt.close()



# =============================================================================
# PASO 6 — INTEGRIDAD
# =============================================================================
titulo("PASO 6 — NADA SE MODIFICO")

huella_despues = {p.name: md5(p) for p in (RUTA_MODELO, RUTA_METRICAS)}
for nombre in huella_antes:
    check(huella_antes[nombre] == huella_despues[nombre], f"{nombre} intacto")
check(len(tabla) == len(COLS), "tabla con una fila por variable del modelo")
check(tabla.drop(columns=["variable", "nombre"]).notna().all().all(),
      "tabla sin valores faltantes")
