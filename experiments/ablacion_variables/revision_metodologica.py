# -*- coding: utf-8 -*-
"""
revision_metodologica.py — Revision antes de sustituir H (10 vars) por E (8 vars)
Proyecto: Give Me Some Credit — Probability of Default

EXPERIMENTO AISLADO: solo escribe en experiments/ablacion_variables/resultados/.
Solo usa las 120,000 filas de entrenamiento. Ni validacion ni Kaggle.

  1. Cortes WoE: ¿se eligieron viendo el target de los datos de la CV?
     -> CV con cortes ChiMerge derivados DENTRO de cada particion.
  2. Multiples comparaciones: Holm sobre las 56 pruebas de ablacion.py y
     replica confirmatoria de A vs E con particiones nuevas (E se eligio a
     posteriori, despues de ver los resultados).
  3. MonthlyIncome y capacidad de pago: ¿sigue el modelo E la informacion
     del ingreso? Calibracion por grupo de ingreso, fuera de muestra.
  4. No inferioridad de E frente a A con margenes fijados ANTES de la replica.
  5. Seleccion: E como candidato formal en el procedimiento de 03, en train.

Se ejecuta con:  python experiments/ablacion_variables/revision_metodologica.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss)

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
from src import preprocessing as pp
from src import woe as woe_mod
from src.artifacts import ruta_datos
from src.evaluation import (ks_statistic, error_calibracion, umbral_optimo,
                            signos_incoherentes)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

SALIDA = AQUI / "resultados"
SEMILLA = 42
SEMILLA_REPLICA = 20261101          # particiones nuevas, nunca vistas
TARGET = pp.TARGET
PARAMS_LR = dict(max_iter=2000, tol=1e-8, C=1.0, random_state=SEMILLA)
MARGEN, PERDIDA = 800, 10_000 * 0.60
UMBRAL = umbral_optimo(MARGEN, PERDIDA)

# --- Margenes de no inferioridad (E - A), fijados ANTES de la replica --------
# Cada uno es una diferencia que el negocio NO consideraria material:
#   AUC   0.001 : 28% de la desv. del AUC entre folds (0.0036); 1/8 de la
#                 distancia LogReg-boosters (0.008); 40% de lo que se acepto
#                 perder en H1 (0.0026) por una razon de gobierno.
#   AP    0.002 : 0.5% relativo sobre 0.38; 1/10 de la distancia LogReg-XGB.
#   Calib 0.05pp: el error se mide con resolucion de 0.01 pp por decil; 0.05
#                 es ~7% del error actual (0.73 pp).
#   Benef $1.00 : por solicitud = 0.2% de ~$513; $30,000 en 30,000 solicitudes.
#   LogLoss 0.0002: 40% de lo que cuesta quitar RealEstateLoans (0.00042),
#                 la unica eliminacion que se juzgo material.
# Direccion: AUC, AP, Benef -> mayor es mejor; Calib, LogLoss -> menor es mejor.
MARGENES = {"AUC": (0.001, +1), "AP": (0.002, +1), "Benef_sol": (1.00, +1),
            "Calib_pp": (0.05, -1), "LogLoss": (0.0002, -1)}

ORIGINALES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio_limpio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
COLS_A = [f"woe_{v}" for v in ORIGINALES]
FUERA_E = {"woe_MonthlyIncome", "woe_NumberOfOpenCreditLinesAndLoans"}
COLS_E = [c for c in COLS_A if c not in FUERA_E]
ESC = {"A": COLS_A, "E": COLS_E}


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


def nb_se(d, n_tr, n_te):
    d = np.asarray(d, dtype=float)
    return np.sqrt(np.var(d, ddof=1) * (1 / len(d) + n_te / n_tr))


def cv_A_vs_E(X, y, semilla, bins_por_fold=False, guardar_oof=False):
    """CV 5x5 pareada. bins_por_fold=True -> cortes ChiMerge dentro del fold."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=semilla)
    filas, oof = [], {k: np.full(len(y), np.nan) for k in ESC}
    for k, (itr, ite) in enumerate(rskf.split(X, y)):
        Xa, Xb = X.iloc[itr], X.iloc[ite]
        ya, yb = y.iloc[itr], y.iloc[ite]
        bins = None
        if bins_por_fold:
            bins = {v: woe_mod.chimerge(Xa[v], ya,
                                        special_values=[0] if v.startswith("Number") else None)
                    for v in ORIGINALES}
        tablas = woe_mod.aprender_woe(Xa[ORIGINALES], ya, bins=bins)
        Wa = woe_mod.aplicar_woe(Xa[ORIGINALES], tablas, bins=bins)[COLS_A]
        Wb = woe_mod.aplicar_woe(Xb[ORIGINALES], tablas, bins=bins)[COLS_A]
        for nombre, cols in ESC.items():
            m = LogisticRegression(**PARAMS_LR).fit(Wa[cols], ya)
            p = m.predict_proba(Wb[cols])[:, 1]
            filas.append({"fold": k, "esc": nombre, **metricas(yb.to_numpy(), p)})
            if guardar_oof and k < 5:                   # primera repeticion = OOF
                oof[nombre][ite] = p
    return pd.DataFrame(filas), oof


def no_inferioridad(cv, etiqueta, n_tr, n_te):
    a = cv[cv.esc == "A"].set_index("fold")
    e = cv[cv.esc == "E"].set_index("fold")
    filas = []
    q90 = stats.t.ppf(0.95, cv.fold.nunique() - 1)     # IC 90% = test unilateral 5%
    for mtr in ["AUC", "AP", "KS", "LogLoss", "Brier", "Calib_pp", "Benef_sol"]:
        d = (e[mtr] - a[mtr]).to_numpy()
        se = nb_se(d, n_tr, n_te)
        lo, hi = d.mean() - q90 * se, d.mean() + q90 * se
        fila = {"evidencia": etiqueta, "metrica": mtr, "A": a[mtr].mean(), "E": e[mtr].mean(),
                "dif_E_menos_A": d.mean(), "IC90_lo": lo, "IC90_hi": hi}
        if mtr in MARGENES:
            delta, signo = MARGENES[mtr]
            fila["margen"] = delta
            fila["no_inferior"] = bool(lo > -delta) if signo > 0 else bool(hi < delta)
            fila["equivalente"] = bool(lo > -delta and hi < delta)
        filas.append(fila)
    return pd.DataFrame(filas)


# =============================================================================
# DATOS (mismo split que 02_preprocessing.py; validacion NO se usa)
# =============================================================================
raw = pd.read_csv(ruta_datos(RAIZ) / "cs-training.csv", index_col=0)
raw.index.name = "row_id"
X_tr_raw, _, y_tr, _ = train_test_split(raw.drop(columns=[TARGET]), raw[TARGET],
                                        test_size=0.20, stratify=raw[TARGET],
                                        random_state=SEMILLA)
X_tr = pp.crear_features(pp.limpiar(X_tr_raw))
ref = pd.read_parquet(RAIZ / "datasets" / "processed" / "woe" / "train.parquet")
assert X_tr.index.equals(ref.index)
n_te = len(X_tr) // 5
n_tr = len(X_tr) - n_te


# =============================================================================
# 1. ORIGEN DE LOS CORTES Y SESGO
# =============================================================================
titulo("1. ORIGEN DE LOS CORTES WoE")

full = pp.crear_features(pp.limpiar(raw.drop(columns=[TARGET])))
forense = []
for nom, X, y in [("150k (fichero completo)", full, raw[TARGET]), ("120k (train)", X_tr, y_tr)]:
    r = y.groupby(pd.cut(X.RevolvingUtilizationOfUnsecuredLines, [-0.1, .5, .9, 1.0, 1.2]),
                  observed=True).mean() * 100
    a = y.groupby(pd.qcut(X.age, 10), observed=True).mean() * 100
    forense.append({"datos": nom, "RevUtil_0-0.5": r.iloc[0], "RevUtil_0.5-0.9": r.iloc[1],
                    "RevUtil_0.9-1.0": r.iloc[2], "RevUtil_1.0-1.2": r.iloc[3],
                    "age_decil1": a.iloc[0], "age_decil10": a.iloc[-1]})
forense.append({"datos": "comentario en src/woe.py", "RevUtil_0-0.5": 2.74,
                "RevUtil_0.5-0.9": 12.20, "RevUtil_0.9-1.0": 19.31, "RevUtil_1.0-1.2": 38.23,
                "age_decil1": 11.09, "age_decil10": 2.19})
print(pd.DataFrame(forense).round(2).to_string(index=False))
print("\n  -> Los cortes se escribieron mirando tasas de default de los 120k de train:")
print("     el mismo conjunto de la CV. Validacion no intervino.")

cv_chi, _ = cv_A_vs_E(X_tr, y_tr, SEMILLA, bins_por_fold=True)
cv_man, oof = cv_A_vs_E(X_tr, y_tr, SEMILLA, guardar_oof=True)
cv_rep, _ = cv_A_vs_E(X_tr, y_tr, SEMILLA_REPLICA)
for nombre, d in [("cortes manuales", cv_man), ("ChiMerge por particion", cv_chi)]:
    m = d.groupby("esc")[["AUC", "AP", "LogLoss"]].mean()
    print(f"\n  {nombre:<24} AUC A={m.loc['A','AUC']:.5f} E={m.loc['E','AUC']:.5f} | "
          f"AP A={m.loc['A','AP']:.5f} E={m.loc['E','AP']:.5f}")
optim = cv_man[cv_man.esc == "A"].AUC.mean() - cv_chi[cv_chi.esc == "A"].AUC.mean()
print(f"\n  AUC de A con cortes manuales - con cortes por particion: {optim:+.5f}")
print("  (cota del optimismo de elegir los cortes viendo todo train; incluye tambien")
print("   la diferencia de calidad entre el binning manual y el automatico)")


# =============================================================================
# 2. MULTIPLES COMPARACIONES
# =============================================================================
titulo("2. MULTIPLES COMPARACIONES (Holm sobre las 56 pruebas de ablacion.py)")

par = pd.read_csv(SALIDA / "comparaciones_pareadas.csv")
pruebas = []
for _, r in par.iterrows():
    for mtr in ["AUC", "AP", "KS", "LogLoss", "Brier", "Calib_pp", "Benef_sol", "gap_AUC"]:
        pruebas.append({"escenario": r.escenario, "metrica": mtr, "dif": r[f"d{mtr}"],
                        "p": r[f"d{mtr}_p"]})
pruebas = pd.DataFrame(pruebas).sort_values("p").reset_index(drop=True)
m = len(pruebas)
pruebas["p_holm"] = np.minimum(1, np.maximum.accumulate(pruebas["p"] * (m - np.arange(m))))
pruebas["sig_sin_corregir"] = pruebas["p"] < 0.05
pruebas["sig_holm"] = pruebas["p_holm"] < 0.05
pruebas.to_csv(SALIDA / "revision_holm.csv", index=False)
print(pruebas[pruebas.sig_sin_corregir].round(5).to_string(index=False))
print(f"\n  {m} pruebas | significativas sin corregir: {pruebas.sig_sin_corregir.sum()} | "
      f"tras Holm: {pruebas.sig_holm.sum()}")


# =============================================================================
# 3. MonthlyIncome Y CAPACIDAD DE PAGO
# =============================================================================
titulo("3. ¿SIGUE E LA INFORMACION DEL INGRESO? (OOF, 1a repeticion de la CV)")

sin_ingreso = X_tr["DebtRatio_limpio"].isna()
invalido = (X_tr.flag_income_faltante | X_tr.flag_income_cero).astype(bool)
print(f"  DebtRatio FALTANTE <=> ingreso faltante o cero: {(sin_ingreso == invalido).all()}"
      f"  ({invalido.sum():,} filas)")
print(f"  DebtRatio_limpio = deuda / ingreso: el ingreso entra como denominador")

grupo = pd.qcut(X_tr["MonthlyIncome"], 10, labels=[f"D{i}" for i in range(1, 11)]
                ).astype(object).where(~X_tr["MonthlyIncome"].isna(), "sin ingreso")
cal = pd.DataFrame({"grupo": grupo.to_numpy(), "y": y_tr.to_numpy(),
                    "PD_A": oof["A"], "PD_E": oof["E"]})
cal = cal.groupby("grupo").agg(n=("y", "size"), default_real=("y", "mean"),
                               PD_A=("PD_A", "mean"), PD_E=("PD_E", "mean")) * \
      pd.Series({"n": 1, "default_real": 100, "PD_A": 100, "PD_E": 100})
cal["error_A_pp"] = cal.PD_A - cal.default_real
cal["error_E_pp"] = cal.PD_E - cal.default_real
orden = [f"D{i}" for i in range(1, 11)] + ["sin ingreso"]
cal = cal.loc[orden]
cal.to_csv(SALIDA / "revision_calibracion_por_ingreso.csv")
print(cal.round(3).to_string())
print(f"\n  Error absoluto medio por grupo de ingreso: A {cal.error_A_pp.abs().mean():.3f} pp"
      f" | E {cal.error_E_pp.abs().mean():.3f} pp")
print(f"  Error maximo:                            A {cal.error_A_pp.abs().max():.3f} pp"
      f" | E {cal.error_E_pp.abs().max():.3f} pp")
rho_A = stats.spearmanr(cal.loc[orden[:10], "PD_A"], cal.loc[orden[:10], "default_real"])[0]
rho_E = stats.spearmanr(cal.loc[orden[:10], "PD_E"], cal.loc[orden[:10], "default_real"])[0]
print(f"  Spearman(PD media, default real) en los deciles de ingreso: A {rho_A:.3f} | E {rho_E:.3f}")


# =============================================================================
# 4. NO INFERIORIDAD DE E FRENTE A A
# =============================================================================
titulo("4. NO INFERIORIDAD (IC 90% Nadeau-Bengio = prueba unilateral al 5%)")

ni = pd.concat([
    no_inferioridad(cv_man, "CV original (seed 42, cortes manuales)", n_tr, n_te),
    no_inferioridad(cv_rep, f"Replica confirmatoria (seed {SEMILLA_REPLICA})", n_tr, n_te),
    no_inferioridad(cv_chi, "Cortes ChiMerge dentro de cada particion", n_tr, n_te),
], ignore_index=True)
ni.to_csv(SALIDA / "revision_no_inferioridad.csv", index=False)
for ev, g in ni.groupby("evidencia", sort=False):
    print(f"\n  {ev}")
    print(g.drop(columns="evidencia").to_string(index=False, float_format=lambda v: f"{v:.6f}"))
conf = ni[ni.evidencia.str.startswith("Replica") & ni.margen.notna()]
print(f"\n  REPLICA CONFIRMATORIA: no inferior en las {len(conf)} metricas con margen: "
      f"{bool(conf.no_inferior.all())}  (prueba de interseccion-union: exige todas,")
print("  asi que no necesita correccion por multiplicidad)")


# =============================================================================
# 5. E COMO CANDIDATO FORMAL EN LA SELECCION DE 03 (solo train)
# =============================================================================
titulo("5. SELECCION DE 03 CON E COMO CANDIDATO (ajuste/seleccion dentro de train)")

W = ref.drop(columns=[TARGET])
X_fit, X_sel, y_fit, y_sel = train_test_split(W, ref[TARGET], test_size=0.20,
                                              stratify=ref[TARGET], random_state=SEMILLA)
FLAGS = [c for c in W.columns if c.startswith("flag_")]
DERIV = [c for c in W.columns if c not in COLS_A + FLAGS]
sin = lambda cols: [c for c in W.columns if c not in cols]
FLAGS_CAL = ["flag_income_faltante", "flag_income_cero", "flag_dependents_faltante",
             "flag_registro_invalido", "flag_revutil_invalido", "flag_edad_invalida"]
CAND = {
    "A: todas": list(W.columns),
    "B: sin flag_algun, con total_atrasos": sin(["woe_peor_atraso", "flag_algun_atraso"]),
    "C: sin flag_algun, con peor_atraso": sin(["woe_total_atrasos", "flag_algun_atraso"]),
    "D: sin derivadas de morosidad": sin(["woe_total_atrasos", "woe_peor_atraso",
                                          "flag_algun_atraso"]),
    "E: originales + todos los flags": COLS_A + FLAGS,
    "F: sin flags de calidad de dato": sin(FLAGS_CAL),
    "G: originales + flag_algun_atraso": COLS_A + ["flag_algun_atraso"],
    "H: solo las 10 originales": COLS_A,
    "I: 8 originales (sin MonthlyIncome ni OpenCreditLines)": COLS_E,
}


def vif_max(Xc):
    out = []
    for c in Xc.columns:
        otras = Xc.drop(columns=c)
        r2 = LinearRegression().fit(otras, Xc[c]).score(otras, Xc[c])
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return max(out)


filas = []
p_sel = {}
for nombre, cols in CAND.items():
    m = LogisticRegression(**PARAMS_LR).fit(X_fit[cols], y_fit)
    p_sel[nombre] = m.predict_proba(X_sel[cols])[:, 1]
    filas.append({"subconjunto": nombre, "n_vars": len(cols),
                  "AUC_seleccion": roc_auc_score(y_sel, p_sel[nombre]),
                  "AP_seleccion": average_precision_score(y_sel, p_sel[nombre]),
                  "VIF_max": vif_max(X_fit[cols]),
                  "incoherentes": len(signos_incoherentes(m.coef_[0], cols, X_fit, y_fit))})
sel = pd.DataFrame(filas)
print(sel.round(5).to_string(index=False))

validos = sel[sel.incoherentes == 0]
# Criterio ACTUAL de 03: max AUC_seleccion, desempate por VIF
actual = validos.sort_values(["AUC_seleccion", "VIF_max"], ascending=[False, True]).iloc[0]
# Criterio PROPUESTO: parsimonia bajo no inferioridad. Entre los coherentes, el
# de MENOS variables cuyo AUC de seleccion no quede mas de MARGEN_AUC por
# debajo del mejor; desempate por AUC.
mejor = validos.AUC_seleccion.max()
elegibles = validos[validos.AUC_seleccion >= mejor - MARGENES["AUC"][0]]
propuesto = elegibles.sort_values(["n_vars", "AUC_seleccion"], ascending=[True, False]).iloc[0]
print(f"\n  Criterio ACTUAL   (max AUC_seleccion)              -> {actual.subconjunto}")
print(f"  Criterio PROPUESTO (menos vars, AUC >= mejor - {MARGENES['AUC'][0]}) -> {propuesto.subconjunto}")

# Incertidumbre de la diferencia H vs I en el set de seleccion (bootstrap pareado)
rng = np.random.default_rng(SEMILLA)
ys = y_sel.to_numpy()
pH, pI = p_sel["H: solo las 10 originales"], p_sel["I: 8 originales (sin MonthlyIncome ni OpenCreditLines)"]
boots = []
for _ in range(1000):
    i = rng.integers(0, len(ys), len(ys))
    boots.append(roc_auc_score(ys[i], pI[i]) - roc_auc_score(ys[i], pH[i]))
lo, hi = np.percentile(boots, [5, 95])
print(f"  AUC_sel(I) - AUC_sel(H) = {roc_auc_score(ys, pI) - roc_auc_score(ys, pH):+.5f}  "
      f"IC90% bootstrap pareado [{lo:+.5f}, {hi:+.5f}]")
sel.to_csv(SALIDA / "revision_seleccion_con_E.csv", index=False)
