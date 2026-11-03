# -*- coding: utf-8 -*-
"""
ablacion.py — ¿Sobran variables en el modelo H (LogReg WoE, 10 variables)?
Proyecto: Give Me Some Credit — Probability of Default

EXPERIMENTO AISLADO. Lee los CSV crudos y los artefactos actuales, pero solo
escribe en experiments/ablacion_variables/resultados/. No toca modelos,
preprocesadores, datasets, metricas ni graficas del pipeline.

Diseno
  - Mismo split que 02_preprocessing.py (seed 42): la CV se hace SOLO sobre
    los 120,000 de train. Validacion (30,000) se usa una vez, al final, como
    confirmacion: no interviene en ninguna decision. Test (Kaggle) no se toca.
  - CV estratificada repetida 5x5 = 25 particiones, IDENTICAS para todos los
    escenarios -> comparaciones pareadas.
  - Dentro de cada particion las tablas WoE se aprenden solo con su parte de
    entrenamiento. Los CORTES de los bins son las constantes manuales del
    proyecto (BINS_TODOS), no se re-derivan por particion.
  - Mismos hiperparametros que 03 (tol=1e-8, C=1, lbfgs, max_iter=2000).
  - Incertidumbre: t-test con varianza corregida de Nadeau & Bengio (2003),
    porque las particiones de una CV repetida se solapan.

Se ejecuta con:  python experiments/ablacion_variables/ablacion.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # signo "-" normal en los ejes
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss)

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src import preprocessing as pp
from src import woe as woe_mod
from src.evaluation import ks_statistic, error_calibracion, umbral_optimo
from src.artifacts import ruta_datos

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
TARGET = pp.TARGET
PARAMS_LR = dict(max_iter=2000, tol=1e-8, C=1.0, random_state=SEMILLA)
MARGEN, PERDIDA = 800, 10_000 * 0.60
UMBRAL = umbral_optimo(MARGEN, PERDIDA)
N_SPLITS, N_REPEATS = 5, 5

ORIGINALES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio_limpio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
COLS_A = [f"woe_{v}" for v in ORIGINALES]          # orden exacto del modelo H
CANDIDATAS = ["NumberRealEstateLoansOrLines", "NumberOfDependents",
              "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans"]


def sin(*vs):
    fuera = {f"woe_{v}" for v in vs}
    return [c for c in COLS_A if c not in fuera]


ESCENARIOS = {
    "A: 10 originales": COLS_A,
    "B1: sin RealEstateLoans": sin("NumberRealEstateLoansOrLines"),
    "B2: sin Dependents": sin("NumberOfDependents"),
    "B3: sin MonthlyIncome": sin("MonthlyIncome"),
    "B4: sin OpenCreditLines": sin("NumberOfOpenCreditLinesAndLoans"),
    # C: las 3 de menor caida de AUC por permutacion (03b): rangos 8, 9 y 10
    "C: sin Dep+Income+OpenCredit": sin("NumberOfDependents", "MonthlyIncome",
                                        "NumberOfOpenCreditLinesAndLoans"),
    "D: sin las 4 candidatas": sin(*CANDIDATAS),
    # E (adicional): las 2 con importancia ~0 en AMBOS metodos (|beta*sigma| < 0.02
    # y caida de AUC < 1e-4); Dependents si tiene una caida positiva pequena.
    "E: sin Income+OpenCredit": sin("MonthlyIncome", "NumberOfOpenCreditLinesAndLoans"),
}


def titulo(t):
    print("\n" + "=" * 79 + f"\n{t}\n" + "=" * 79)


def metricas(y, p):
    return {
        "AUC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "KS": ks_statistic(y, p)[0],
        "LogLoss": log_loss(y, p),
        "Brier": brier_score_loss(y, p),
        "Calib_pp": error_calibracion(y, p),
        "Benef_sol": (((y == 0) & (p < UMBRAL)).sum() * MARGEN
                      - ((y == 1) & (p < UMBRAL)).sum() * PERDIDA) / len(y),
    }


def nadeau_bengio(d, n_tr, n_te):
    """Media, IC95% y p-valor del t-test corregido para CV repetida."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + n_te / n_tr)
    se = np.sqrt(var)
    t = d.mean() / se if se > 0 else 0.0
    q = stats.t.ppf(0.975, J - 1)
    return d.mean(), d.mean() - q * se, d.mean() + q * se, 2 * stats.t.sf(abs(t), J - 1)


# =============================================================================
# PASO 1 — DATOS (mismo split que 02_preprocessing.py)
# =============================================================================
titulo("PASO 1 — DATOS")

raw = pd.read_csv(ruta_datos(RAIZ) / "cs-training.csv", index_col=0)
raw.index.name = "row_id"
X_raw, y_raw = raw.drop(columns=[TARGET]), raw[TARGET]
X_tr, X_val, y_tr, y_val = train_test_split(
    X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=SEMILLA)
X_tr = pp.crear_features(pp.limpiar(X_tr))
X_val = pp.crear_features(pp.limpiar(X_val))

ref_tr = pd.read_parquet(RAIZ / "datasets" / "processed" / "woe" / "train.parquet")
ref_val = pd.read_parquet(RAIZ / "datasets" / "processed" / "woe" / "val.parquet")
assert X_tr.index.equals(ref_tr.index) and X_val.index.equals(ref_val.index), \
    "el split no coincide con el del pipeline"

# El WoE aprendido aqui con todo train debe reproducir el parquet del pipeline.
tablas_tr = woe_mod.aprender_woe(X_tr, y_tr)
W_tr = woe_mod.aplicar_woe(X_tr, tablas_tr)[COLS_A]
W_val = woe_mod.aplicar_woe(X_val, tablas_tr)[COLS_A]
assert np.allclose(W_tr, ref_tr[COLS_A]) and np.allclose(W_val, ref_val[COLS_A]), \
    "el WoE reconstruido no coincide con el del pipeline"
print(f"  Train {X_tr.shape[0]:,} | validacion {X_val.shape[0]:,} (solo confirmacion final)")
print("  Split y WoE reconstruidos = pipeline: OK")
print(f"  Umbral de decision PD < {UMBRAL:.4f}  (M={MARGEN}, L={PERDIDA:,.0f})")


# =============================================================================
# PASO 2 — RELEVANCIA DE CADA CANDIDATA (sobre train)
# =============================================================================
titulo("PASO 2 — RELEVANCIA DE LAS CANDIDATAS")

perm = pd.read_csv(RAIZ / "artifacts" / "metrics" / "logistic_interpretabilidad.csv")
perm["v"] = perm["variable"].str[4:]
perm = perm.set_index("v")
corr = W_tr.corr()

relevancia, tablas_bins = [], []
for v in CANDIDATAS:
    col = f"woe_{v}"
    otras = [c for c in COLS_A if c != col]
    r2 = LinearRegression().fit(W_tr[otras], W_tr[col]).score(W_tr[otras], W_tr[col])
    socio = corr[col].drop(col).abs().idxmax()
    # AUC univariado: el WoE ordena en sentido inverso al riesgo
    auc_uni = roc_auc_score(y_tr, -W_tr[col])
    t = tablas_tr[v]
    relevancia.append({
        "variable": v,
        "IV": round(t["iv_parcial"].sum(), 4),
        "n_bins": len(t),
        "AUC_univariado": round(auc_uni, 4),
        "perm_dAUC": perm.loc[v, "auc_drop_mean"], "perm_dAUC_sd": perm.loc[v, "auc_drop_std"],
        "perm_dAP": perm.loc[v, "ap_drop_mean"], "perm_dAP_sd": perm.loc[v, "ap_drop_std"],
        "beta": perm.loc[v, "beta"], "beta_std": perm.loc[v, "beta_std"],
        "R2_vs_otras9": round(r2, 4),
        "corr_max": round(corr.loc[col, socio], 3), "socio": socio[4:],
    })
    tb = t[["n", "pct_poblacion", "default_rate_pct", "woe", "iv_parcial"]].copy()
    tb.insert(0, "bin", tb.index.astype(str))
    tb.insert(0, "variable", v)
    tablas_bins.append(tb.reset_index(drop=True))

relevancia = pd.DataFrame(relevancia)
tablas_bins = pd.concat(tablas_bins, ignore_index=True)
print(relevancia.to_string(index=False))
print("\n  Tablas WoE (train):")
print(tablas_bins.round(4).to_string(index=False))


# =============================================================================
# PASO 3 — CV REPETIDA 5x5 CON WoE DENTRO DE CADA PARTICION
# =============================================================================
titulo(f"PASO 3 — CV ESTRATIFICADA {N_SPLITS}x{N_REPEATS} (WoE re-aprendido por particion)")

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEMILLA)
filas, coefs, iv_fold = [], [], []
for k, (itr, ite) in enumerate(rskf.split(X_tr, y_tr)):
    Xa, Xb = X_tr.iloc[itr], X_tr.iloc[ite]
    ya, yb = y_tr.iloc[itr].to_numpy(), y_tr.iloc[ite].to_numpy()
    tablas = woe_mod.aprender_woe(Xa, pd.Series(ya, index=Xa.index))   # solo la parte de ajuste
    Wa = woe_mod.aplicar_woe(Xa, tablas)[COLS_A]
    Wb = woe_mod.aplicar_woe(Xb, tablas)[COLS_A]
    iv_fold.append({v: tablas[v]["iv_parcial"].sum() for v in CANDIDATAS})
    p_A = None
    for nombre, cols in ESCENARIOS.items():
        m = LogisticRegression(**PARAMS_LR).fit(Wa[cols], ya)
        pa, pb = m.predict_proba(Wa[cols])[:, 1], m.predict_proba(Wb[cols])[:, 1]
        if p_A is None:
            p_A = pb
        mt, mv = metricas(ya, pa), metricas(yb, pb)
        filas.append({"fold": k, "escenario": nombre, "n_iter": int(m.n_iter_[0]),
                      **{f"{key}": val for key, val in mv.items()},
                      "AUC_train": mt["AUC"], "LogLoss_train": mt["LogLoss"],
                      "decisiones_cambiadas_pct":
                          float(((pb < UMBRAL) != (p_A < UMBRAL)).mean() * 100)})
        coefs.append({"fold": k, "escenario": nombre, "intercepto": m.intercept_[0],
                      **dict(zip(cols, m.coef_[0]))})
    print(f"  particion {k + 1:>2}/{N_SPLITS * N_REPEATS} lista")

cv = pd.DataFrame(filas)
cv["gap_AUC"] = cv["AUC_train"] - cv["AUC"]
cv["gap_LogLoss"] = cv["LogLoss"] - cv["LogLoss_train"]
coefs = pd.DataFrame(coefs)
cv.to_csv(SALIDA / "cv_por_particion.csv", index=False)
coefs.to_csv(SALIDA / "coeficientes_por_particion.csv", index=False)
n_te = len(X_tr) // N_SPLITS
n_tr = len(X_tr) - n_te
assert cv["n_iter"].max() < PARAMS_LR["max_iter"], "algun ajuste no convergio"


# =============================================================================
# PASO 4 — RESUMEN Y COMPARACIONES PAREADAS CONTRA A
# =============================================================================
titulo("PASO 4 — RESUMEN Y COMPARACIONES PAREADAS (Nadeau-Bengio)")

METR = ["AUC", "AP", "KS", "LogLoss", "Brier", "Calib_pp", "Benef_sol",
        "gap_AUC", "gap_LogLoss", "decisiones_cambiadas_pct"]
resumen = cv.groupby("escenario", sort=False)[METR].agg(["mean", "std"])
resumen.columns = [f"{a}_{b}" for a, b in resumen.columns]
resumen.insert(0, "n_vars", [len(ESCENARIOS[e]) for e in resumen.index])
print(resumen[["n_vars"] + [f"{m}_mean" for m in METR]].round(5).to_string())

base = cv[cv.escenario == "A: 10 originales"].set_index("fold")
pareadas = []
for nombre in list(ESCENARIOS)[1:]:
    otro = cv[cv.escenario == nombre].set_index("fold")
    fila = {"escenario": nombre}
    for mtr in ["AUC", "AP", "KS", "LogLoss", "Brier", "Calib_pp", "Benef_sol", "gap_AUC"]:
        d = (otro[mtr] - base[mtr]).to_numpy()
        media, lo, hi, p = nadeau_bengio(d, n_tr, n_te)
        fila.update({f"d{mtr}": media, f"d{mtr}_lo": lo, f"d{mtr}_hi": hi, f"d{mtr}_p": p,
                     f"d{mtr}_peor_pct": float(
                         (d < 0).mean() * 100 if mtr in ("AUC", "AP", "KS", "Benef_sol")
                         else (d > 0).mean() * 100)})
    pareadas.append(fila)
pareadas = pd.DataFrame(pareadas)
pareadas.to_csv(SALIDA / "comparaciones_pareadas.csv", index=False)
resumen.to_csv(SALIDA / "resumen_cv.csv")

for mtr, fmt in [("AUC", ".5f"), ("AP", ".5f"), ("KS", ".5f"), ("LogLoss", ".6f"),
                 ("Brier", ".6f"), ("Calib_pp", ".3f"), ("Benef_sol", ".2f")]:
    print(f"\n  Δ{mtr} vs A  (media [IC95% corregido]  p  | % particiones en que empeora)")
    for _, r in pareadas.iterrows():
        print(f"    {r.escenario:<30} {r[f'd{mtr}']:+{fmt}} [{r[f'd{mtr}_lo']:+{fmt}}, "
              f"{r[f'd{mtr}_hi']:+{fmt}}]  p={r[f'd{mtr}_p']:.3f}  | {r[f'd{mtr}_peor_pct']:.0f}%")


# =============================================================================
# PASO 5 — ESTABILIDAD DE COEFICIENTES
# =============================================================================
titulo("PASO 5 — ESTABILIDAD DE LOS COEFICIENTES (25 particiones)")

estab = []
for nombre in ESCENARIOS:
    c = coefs[coefs.escenario == nombre].drop(columns=["fold", "escenario"]).dropna(axis=1)
    for col in c.columns:
        if col == "intercepto":
            continue
        estab.append({"escenario": nombre, "variable": col[4:],
                      "beta_media": c[col].mean(), "beta_sd": c[col].std(ddof=1),
                      "cv_abs": c[col].std(ddof=1) / abs(c[col].mean()),
                      "pct_negativos": (c[col] < 0).mean() * 100})
estab = pd.DataFrame(estab)
estab.to_csv(SALIDA / "estabilidad_coeficientes.csv", index=False)
print(estab[estab.escenario == "A: 10 originales"].round(4).to_string(index=False))

# ¿Quitar variables estabiliza las que quedan? Comparamos la sd de las 6 comunes.
comunes = [c[4:] for c in ESCENARIOS["D: sin las 4 candidatas"]]
sd_cmp = estab[estab.variable.isin(comunes)].pivot(index="variable", columns="escenario",
                                                    values="beta_sd")[list(ESCENARIOS)]
print("\n  sd de beta de las 6 variables comunes, por escenario:")
print(sd_cmp.round(4).to_string())

ivf = pd.DataFrame(iv_fold)
print("\n  IV de las candidatas entre particiones (media ± sd):")
for v in CANDIDATAS:
    print(f"    {v:<34} {ivf[v].mean():.4f} ± {ivf[v].std(ddof=1):.4f}")


# =============================================================================
# PASO 6 — CONFIRMACION EN VALIDACION (no decide nada)
# =============================================================================
titulo("PASO 6 — CONFIRMACION EN VALIDACION (30,000; WoE aprendido en todo train)")

conf, p_val_A = [], None
for nombre, cols in ESCENARIOS.items():
    m = LogisticRegression(**PARAMS_LR).fit(W_tr[cols], y_tr)
    p = m.predict_proba(W_val[cols])[:, 1]
    if p_val_A is None:
        p_val_A = p
    yv = y_val.to_numpy()
    aprob_A, aprob = p_val_A < UMBRAL, p < UMBRAL
    conf.append({"escenario": nombre, **metricas(yv, p),
                 "aprobados": int(aprob.sum()),
                 "dec_cambiadas": int((aprob != aprob_A).sum()),
                 "nuevos_aprobados_morosos": int((aprob & ~aprob_A & (yv == 1)).sum()),
                 "nuevos_rechazados_morosos": int((~aprob & aprob_A & (yv == 1)).sum()),
                 "max_dPD_vs_A": float(np.abs(p - p_val_A).max())})
conf = pd.DataFrame(conf)
conf["Benef_total"] = conf["Benef_sol"] * len(y_val)
conf.to_csv(SALIDA / "confirmacion_validacion.csv", index=False)
print(conf.round(5).to_string(index=False))


# =============================================================================
# PASO 7 — GRAFICA: diferencias pareadas con IC
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
orden = pareadas["escenario"].tolist()[::-1]
for ax, (mtr, nombre_m, escala) in zip(axes, [
        ("AUC", "Δ AUC (×1e3)", 1e3), ("AP", "Δ Average Precision (×1e3)", 1e3),
        ("LogLoss", "Δ Log Loss (×1e3)", 1e3), ("Benef_sol", "Δ beneficio ($/solicitud)", 1)]):
    r = pareadas.set_index("escenario").loc[orden]
    m_, lo, hi = r[f"d{mtr}"] * escala, r[f"d{mtr}_lo"] * escala, r[f"d{mtr}_hi"] * escala
    malo = (hi < 0) if mtr != "LogLoss" else (lo > 0)
    ax.errorbar(m_, range(len(r)), xerr=[m_ - lo, hi - m_], fmt="o",
                color="#3a4f7a", ecolor="#3a4f7a", capsize=3)
    for i, b in enumerate(malo.to_numpy()):
        if b:
            ax.errorbar(m_.iloc[i], i, xerr=[[m_.iloc[i] - lo.iloc[i]], [hi.iloc[i] - m_.iloc[i]]],
                        fmt="o", color="#a8392f", ecolor="#a8392f", capsize=3, zorder=5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(nombre_m, fontsize=10)
    ax.grid(alpha=0.3, axis="x")
axes[0].set_yticks(range(len(orden)))
axes[0].set_yticklabels(orden, fontsize=8)
fig.suptitle("Ablación vs modelo A: CV 5×5, IC 95% de Nadeau–Bengio "
             "(rojo = deterioro significativo)", fontsize=11)
fig.text(0.01, 0.01, "Log Loss: positivo = peor. Resto: negativo = peor.",
         fontsize=8, color="dimgray")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
plt.savefig(SALIDA / "ablacion_diferencias_pareadas.png", dpi=130)
plt.close()

relevancia.to_csv(SALIDA / "relevancia_candidatas.csv", index=False)
tablas_bins.to_csv(SALIDA / "tablas_woe_candidatas.csv", index=False)
import sklearn, platform
(SALIDA / "config.json").write_text(json.dumps({
    "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS}, random_state={SEMILLA}) sobre train",
    "params_lr": PARAMS_LR, "umbral": UMBRAL, "margen": MARGEN, "perdida": PERDIDA,
    "escenarios": {k: [c[4:] for c in v] for k, v in ESCENARIOS.items()},
    "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                  "numpy": np.__version__, "pandas": pd.__version__},
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Resultados en {SALIDA.relative_to(RAIZ)}")
