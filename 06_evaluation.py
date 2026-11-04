# -*- coding: utf-8 -*-
"""
06_evaluation.py: Comparacion final, eleccion del modelo y submission
Proyecto: Give Me Some Credit: Probability of Default

Tres cosas:
  1. Comparar los tres modelos con criterios MULTIPLES, no solo AUC.
  2. Elegir uno de forma razonada y dejar el razonamiento por escrito.
  3. Reentrenarlo con las 150,000 filas etiquetadas y predecir sobre
     cs-test.csv en el formato de sampleEntry.csv.

POR QUE SE REENTRENA CON TODO AL FINAL
  Las 30,000 filas de validacion se apartan para medir sin sesgo. Una vez
  tomadas las decisiones (variables, hiperparametros, numero de arboles), el
  modelo final las usa tambien para entrenar.

ELECCION DEL MODELO
  Se hace con una CV 5x5 dentro de train (src/seleccion.py); validacion solo
  evalua al modelo ya elegido. Esas filas se examinaron durante el desarrollo,
  asi que no son independientes de todas las decisiones historicas
  (docs/04_modelado.md §4.7.2).

SALIDAS
  artifacts/metrics/modelo_elegido.json        <- lo leen 07 y predict.py
  artifacts/models/final/<modelo>_final.joblib <- modelo reentrenado (150k)
  artifacts/preprocessors/preprocessor_final.joblib (emparejado con el anterior)

Se ejecuta con:  python 06_evaluation.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # signo "-" normal en los ejes

from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve)

sys.path.insert(0, str(Path(__file__).parent))
from src import preprocessing as pp
from src import woe as woe_mod
from src import artifacts
from src import seleccion
from src.evaluation import (ks_statistic, gini, error_calibracion,
                            umbral_optimo, beneficio, validar_submission,
                            diferencia_pareada_cv)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
DATA = artifacts.ruta_datos(AQUI)
PROC = AQUI / "datasets" / "processed"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
SUBMISSIONS = AQUI / "artifacts" / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"

# Supuestos economicos (los mismos de 07_business_evaluation.py)
MARGEN, PERDIDA = 800, 6000
UMBRAL_NEGOCIO = MARGEN / (MARGEN + PERDIDA)


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


# =============================================================================
# PASO 1: CARGA
# =============================================================================
titulo("PASO 1: MODELOS Y DATOS")

RAMA = {"logistic_v1": "woe", "xgboost_v1": "trees", "lightgbm_v1": "trees"}
NOMBRE = {"logistic_v1": "LogReg (WoE)", "xgboost_v1": "XGBoost", "lightgbm_v1": "LightGBM"}

modelos = {}
for stem, rama in RAMA.items():
    f = MODELOS / f"{stem}.joblib"
    if f.exists():
        modelos[stem] = joblib.load(f)
        print(f"  {NOMBRE[stem]:<14} rama={rama:<6} "
              f"{len(modelos[stem]['columnas'])} variables")

if not modelos:
    sys.exit("No hay modelos entrenados. Ejecuta 03, 04 y 05 primero.")

val = {r: pd.read_parquet(PROC / r / "val.parquet") for r in set(RAMA.values())}
y_val = val["trees"][TARGET]

pred = {}
for stem, art in modelos.items():
    X = val[RAMA[stem]].drop(columns=[TARGET])
    pred[stem] = art["modelo"].predict_proba(X[art["columnas"]])[:, 1]

# --- Ensemble: promedio de las probabilidades -------------------------------
# Promedio simple de probabilidades: reduce el error si los modelos se
# equivocan en casos distintos.
if len(pred) == 3:
    pred["ensemble"] = np.mean(list(pred.values()), axis=0)
    NOMBRE["ensemble"] = "Ensemble (promedio)"

print(f"\n  Validacion: {len(y_val):,} personas | {int(y_val.sum()):,} morosos")


# =============================================================================
# PASO 2: COMPARACION MULTI-CRITERIO
# =============================================================================
titulo("PASO 2: COMPARACION EN VALIDACION")

cv_tabla = {}
f_cv = METRICAS / "comparacion_modelos_cv.csv"
if f_cv.exists():
    cv_tabla = pd.read_csv(f_cv).set_index("modelo").to_dict("index")

filas = []
for stem, p in pred.items():
    a = roc_auc_score(y_val, p)
    ks, _ = ks_statistic(y_val, p)
    cvi = cv_tabla.get(NOMBRE[stem], {})
    filas.append({
        "modelo": NOMBRE[stem],
        "AUC_val": round(a, 5),
        "Gini": round(gini(a), 4),
        "KS": round(ks, 5),
        "AP": round(average_precision_score(y_val, p), 4),
        "calib_pp": round(error_calibracion(y_val, p), 3),
        "AUC_cv": cvi.get("auc_cv_media", np.nan),
        "cv_desv": cvi.get("auc_cv_desv", np.nan),
        "beneficio_$": beneficio(p, y_val, UMBRAL_NEGOCIO, MARGEN, PERDIDA)["beneficio"],
    })

comp = pd.DataFrame(filas).sort_values("AUC_val", ascending=False).reset_index(drop=True)
print(comp.to_string(index=False))



# =============================================================================
# PASO 3: LA DECISION NO ES SOLO EL AUC
# =============================================================================
titulo("PASO 3: CRITERIOS DE DECISION")

fila = {r["modelo"]: r for _, r in comp.iterrows()}
NOMBRES_BASE = ["LogReg (WoE)", "XGBoost", "LightGBM"]
criterios = pd.DataFrame([
    {"criterio": "AUC validacion", **{n: f"{fila[n]['AUC_val']:.5f}" for n in NOMBRES_BASE}},
    {"criterio": "AUC CV (media)", **{n: f"{fila[n]['AUC_cv']:.5f}" for n in NOMBRES_BASE}},
    {"criterio": "Calibracion validacion (pp)", **{n: f"{fila[n]['calib_pp']:.3f}" for n in NOMBRES_BASE}},
    {"criterio": "Interpretabilidad", "LogReg (WoE)": "scorecard", "XGBoost": "requiere SHAP",
     "LightGBM": "requiere SHAP"},
])
print(criterios.to_string(index=False))

# Diferencias emparejadas sobre los folds de la CV de 05 (mismo calculo que su
# grafica). Si 05 no se ha ejecutado, se omiten.
f_folds = METRICAS / "comparacion_modelos_cv_folds.csv"
pares_cv = {}
if f_folds.exists():
    folds = pd.read_csv(f_folds, index_col="fold")
    n_lab = len(y_val) + len(pd.read_parquet(PROC / "trees" / "train.parquet"))
    k = len(folds)
    print(f"\n  Diferencias de AUC emparejadas fold a fold (CV de 05, IC 95% Nadeau-Bengio):")
    for a, b in [("XGBoost", "LogReg (WoE)"), ("LightGBM", "LogReg (WoE)"), ("XGBoost", "LightGBM")]:
        r = diferencia_pareada_cv(folds[a], folds[b], n_train=n_lab * (k - 1) / k, n_test=n_lab / k)
        pares_cv[(a, b)] = r
        print(f"    {a:<9} - {b:<13} {r['diferencia']:+.5f}  [{r['ic_inf']:+.5f}, {r['ic_sup']:+.5f}]")
else:
    print("\n  (sin comparacion_modelos_cv_folds.csv: ejecuta 05 para ver las diferencias pareadas)")


# =============================================================================
# PASO 4: ELECCION DEL MODELO FINAL
# =============================================================================
titulo("PASO 4: MODELO ELEGIDO")

# Regla (docs/04_modelado.md §4.7.2): entre los modelos con AUC medio de CV a
# menos de 0.002 del mejor, el de menor error de calibracion medio. La CV es
# 5x5 sobre las 120,000 filas de train; validacion no interviene. XGBoost y
# LightGBM son estadisticamente indistinguibles: el desempate por calibracion
# es la regla fijada de antemano, no evidencia de superioridad.
raw = pd.read_csv(DATA / "cs-training.csv", index_col=0)
raw.index.name = "row_id"
X_tr_trees = pd.read_parquet(PROC / "trees" / "train.parquet")
X_tr_crudo = pp.crear_features(pp.limpiar(raw.loc[X_tr_trees.index].drop(columns=[TARGET])))
y_tr = X_tr_trees[TARGET]

print("  CV 5x5 sobre train para la eleccion (~1 min)...")
cv_sel = seleccion.cv_en_train(X_tr_crudo, X_tr_trees, y_tr, modelos, UMBRAL_NEGOCIO,
                               MARGEN, PERDIDA)
cv_sel.to_csv(METRICAS / "seleccion_cv_train.csv", index=False)
medias_cv = cv_sel.groupby("modelo")[["AUC", "Calib_pp", "Brier", "Benef_sol"]].mean()
ELEGIDO, elegibles = seleccion.regla_seleccion(medias_cv)
elegido_nombre = NOMBRE[ELEGIDO]

print(f"\n  {'modelo':<14} {'AUC CV':>9} {'Calib CV (pp)':>14} {'Brier CV':>10}")
for stem, r in medias_cv.sort_values("AUC", ascending=False).iterrows():
    marca = "  <- elegido" if stem == ELEGIDO else ("  (elegible)" if stem in elegibles else "")
    print(f"  {NOMBRE[stem]:<14} {r.AUC:>9.5f} {r.Calib_pp:>14.3f} {r.Brier:>10.5f}{marca}")

n_te = len(y_tr) / 5
pares_sel = {}
if len(elegibles) > 1:
    otro = [m for m in elegibles if m != ELEGIDO][0]
    A = cv_sel[cv_sel.modelo == ELEGIDO].set_index("fold")
    B = cv_sel[cv_sel.modelo == otro].set_index("fold")
    print(f"\n  {elegido_nombre} - {NOMBRE[otro]} (IC 95% Nadeau-Bengio):")
    for mtr in ["AUC", "Calib_pp", "Brier"]:
        r = diferencia_pareada_cv(A[mtr], B[mtr], n_train=len(y_tr) - n_te, n_test=n_te)
        pares_sel[mtr] = r
        print(f"    {mtr:<9} {r['diferencia']:+.5f}  [{r['ic_inf']:+.5f}, {r['ic_sup']:+.5f}]")

ref = fila["LogReg (WoE)"]
eleg = fila[elegido_nombre]
print(f"\n  ELEGIDO: {elegido_nombre}. Evaluacion posterior en validacion: AUC {eleg['AUC_val']:.5f},"
      f" {(eleg['beneficio_$'] - ref['beneficio_$']) / len(y_val):+.2f} $/solicitud frente a LogReg.")

artifacts.guardar_json({
    "modelo": ELEGIDO,
    "nombre": elegido_nombre,
    "regla": ("AUC_cv >= max - 0.002; desempate por menor Calib_pp medio "
              "(CV 5x5 sobre las 120,000 filas de train)"),
    "usa_validacion": False,
    "elegibles": elegibles,
    "medias_cv": medias_cv.round(6).reset_index().to_dict("records"),
    "diferencias_elegibles": {k: {kk: round(vv, 6) for kk, vv in v.items()}
                              for k, v in pares_sel.items()},
}, METRICAS / "modelo_elegido.json")


# =============================================================================
# PASO 4B: EXPLICABILIDAD DEL MODELO ELEGIDO (SHAP)
# =============================================================================
titulo("PASO 4B: ¿POR QUE RECHAZA A QUIEN RECHAZA?")

# El problema de elegir un booster: la regresion logistica da un scorecard
# donde cada rango vale unos puntos fijos, y con eso se le explica a un
# cliente rechazado exactamente que le resto. Un booster de ~100 arboles no
# tiene esa tabla.
#
# SHAP lo resuelve: reparte la prediccion de CADA persona entre sus
# variables, de forma que las contribuciones suman exactamente el
# resultado del modelo. Es lo mas parecido a un scorecard que puede tener
# un booster: con una diferencia clave: en el scorecard los puntos son
# los MISMOS para todos; en SHAP cada persona tiene su propio desglose.
try:
    import shap
    _hay_shap = True
except ImportError:
    _hay_shap = False
    print("  shap no esta instalado (pip install shap). Se omite este paso.")

if _hay_shap and RAMA[ELEGIDO] != "trees":
    print("  El modelo elegido es la regresion logistica: su explicacion exacta es")
    print("  el scorecard (08_scorecard.py). TreeSHAP no aplica.")
elif _hay_shap:
    # Se explica el modelo ELEGIDO. Se usa la version entrenada con 120,000
    # filas (la evaluada en validacion), no la reentrenada con 150,000.
    stem_shap = ELEGIDO
    art_s = modelos[stem_shap]
    Xs = val["trees"].drop(columns=[TARGET])[art_s["columnas"]]

    explainer = shap.TreeExplainer(art_s["modelo"])
    muestra = Xs.sample(3000, random_state=42)
    shap_vals = explainer.shap_values(muestra)

    print(f"  Modelo explicado: {NOMBRE[stem_shap]}  ({len(muestra):,} personas)")

    # --- Importancia global: media del |SHAP| ---------------------------------
    imp = pd.DataFrame({
        "variable": art_s["columnas"],
        "impacto_medio": np.abs(shap_vals).mean(axis=0),
    }).sort_values(["impacto_medio", "variable"], ascending=[False, True]).reset_index(drop=True)
    imp["pct"] = (imp["impacto_medio"] / imp["impacto_medio"].sum() * 100).round(2)

    print("\n  Impacto medio de cada variable sobre la prediccion:")
    print(imp.head(10).to_string(index=False))

    # --- Explicacion individual ----------------------------------------------
    p_s = art_s["modelo"].predict_proba(Xs)[:, 1]
    i_peor = int(np.argmax(p_s))
    fila = Xs.iloc[[i_peor]]
    sv = explainer.shap_values(fila)[0]

    det = pd.DataFrame({
        "variable": art_s["columnas"],
        "valor": fila.iloc[0].values,
        "contribucion": sv,
    })
    det = det.reindex(det["contribucion"].abs().sort_values(ascending=False).index)

    print(f"\n  --- EXPLICACION DE UN RECHAZO CONCRETO ---")
    print(f"  Cliente #{Xs.index[i_peor]}  |  PD = {p_s[i_peor]*100:.1f}%  |  "
          f"¿cayo en mora? {'SI' if y_val.iloc[i_peor]==1 else 'NO'}")
    print(f"  Base (log-odds medio de la poblacion): {explainer.expected_value:+.4f}")
    print("\n  " + det.head(8).to_string(index=False).replace("\n", "\n  "))
    print(f"\n  Suma de contribuciones + base = {explainer.expected_value + sv.sum():+.4f}")
    print(f"  (en log-odds; equivale a PD = "
          f"{1/(1+np.exp(-(explainer.expected_value + sv.sum())))*100:.1f}%)")

    # --- Grafica --------------------------------------------------------------
    fig, axes_s = plt.subplots(1, 2, figsize=(15, 5.5))
    t = imp.head(12).iloc[::-1]
    axes_s[0].barh(range(len(t)), t["impacto_medio"], color="#7a3a6a")
    axes_s[0].set_yticks(range(len(t)))
    axes_s[0].set_yticklabels(t["variable"], fontsize=7)
    axes_s[0].set_xlabel("Impacto medio |SHAP| sobre los log-odds")
    axes_s[0].set_title(f"Importancia global: {NOMBRE[stem_shap]}")
    axes_s[0].grid(alpha=0.3, axis="x")

    d2 = det.head(10).iloc[::-1]
    cols_b = ["#a8392f" if v > 0 else "#2b6247" for v in d2["contribucion"]]
    axes_s[1].barh(range(len(d2)), d2["contribucion"], color=cols_b)
    axes_s[1].set_yticks(range(len(d2)))
    axes_s[1].set_yticklabels([f"{v}={x:g}" for v, x in zip(d2["variable"], d2["valor"])],
                              fontsize=7)
    axes_s[1].axvline(0, color="black", lw=0.8)
    axes_s[1].set_xlabel("Contribucion a los log-odds  (rojo = mas riesgo)")
    axes_s[1].set_title(f"Por que se rechaza al cliente #{Xs.index[i_peor]}")
    axes_s[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(PLOTS / "06_shap_explicabilidad.png", dpi=110)
    plt.close()
    imp.to_csv(METRICAS / "shap_importancia.csv", index=False)
    print(f"\n  Guardado: artifacts/plots/06_shap_explicabilidad.png")
    print(f"            artifacts/metrics/shap_importancia.csv")


# =============================================================================
# PASO 5: REENTRENAR CON LAS 150,000 FILAS
# =============================================================================
titulo("PASO 5: REENTRENAMIENTO CON TODOS LOS DATOS ETIQUETADOS")

art = modelos[ELEGIDO]
rama = RAMA[ELEGIDO]
COLS = art["columnas"]

print(f"  Hasta ahora el modelo se entreno con 120,000 filas (u 96,000 con")
print(f"  early stopping). Ya no hace falta esconder la validacion: las")
print(f"  decisiones estan tomadas.")

test_raw = pd.read_csv(DATA / "cs-test.csv", index_col=0)
test_raw.index.name = "row_id"

if rama == "trees":
    tr = pd.read_parquet(PROC / "trees" / "train.parquet")
    va = pd.read_parquet(PROC / "trees" / "val.parquet")
    todo = pd.concat([tr, va])
    X_todo, y_todo = todo[COLS], todo[TARGET]
    X_test = pd.read_parquet(PROC / "trees" / "test.parquet")[COLS]
else:
    # En la rama WoE hay que RE-APRENDER las tablas con las 150,000 filas.
    # Los cortes de los bins son los mismos; lo que cambia es el WoE de cada
    # bin, ahora calculado con mas datos.
    crudo = pd.read_csv(DATA / "cs-training.csv", index_col=0)
    y_todo = crudo[TARGET]
    Xc = pp.crear_features(pp.limpiar(crudo.drop(columns=[TARGET])))
    tablas = woe_mod.aprender_woe(Xc, y_todo)
    X_todo = woe_mod.aplicar_woe(Xc, tablas)[COLS]
    Tc = pp.crear_features(pp.limpiar(test_raw.drop(columns=[TARGET])))
    X_test = woe_mod.aplicar_woe(Tc, tablas)[COLS]
    print("  (rama WoE: tablas re-aprendidas con las 150,000 filas)")

modelo_final = art["modelo"].__class__(**art["modelo"].get_params())

# El numero de arboles ya se fijo con early stopping; al reentrenar con todo
# no hay set de vigilancia, asi que se usa ese numero como constante.
if "n_arboles" in art:
    n = art["n_arboles"]
    modelo_final.set_params(n_estimators=n)
    for p in ("early_stopping_rounds",):
        if p in modelo_final.get_params():
            modelo_final.set_params(**{p: None})
    print(f"  Arboles fijados en {n} (el valor que eligio el early stopping)")

modelo_final.fit(X_todo, y_todo)
print(f"  Reentrenado con {len(X_todo):,} filas ({y_todo.mean()*100:.2f}% de morosos)")

# Se guarda en una subcarpeta para que 07 (que evalua en validacion los
# *_v1.joblib) no lo cargue: este modelo ya vio las filas de validacion.
ruta_final = MODELOS / "final" / f"{ELEGIDO}_final.joblib"
ruta_final.parent.mkdir(parents=True, exist_ok=True)
for viejo in ruta_final.parent.glob("*_final.joblib"):    # solo un modelo final vigente
    if viejo != ruta_final:
        viejo.unlink()
joblib.dump({"modelo": modelo_final, "columnas": COLS, "rama": rama,
             "n_filas_train": len(X_todo), "base": f"{ELEGIDO}.joblib",
             "preprocesador": "preprocessor_final.joblib"}, ruta_final)
print(f"  Modelo final guardado: {ruta_final.relative_to(AQUI)}")

# --- Congelar el preprocesador que corresponde a ESTE modelo ----------------
# El artefacto de 02_preprocessing.py se aprendio con 120,000 filas. Si el
# modelo desplegado se entreno con 150,000, ambos tienen que ir emparejados:
# un modelo y un preprocesador de "epocas" distintas producen puntuaciones
# que nadie puede reproducir despues.
art_pre = artifacts.cargar(AQUI / "artifacts" / "preprocessors" / "preprocessor_v1.joblib")
art_final = artifacts.construir_artefacto(
    tablas_woe=(tablas if rama == "woe" else art_pre["tablas_woe"]),
    columnas_trees=(COLS if rama == "trees" else art_pre["columnas_trees"]),
    columnas_woe=(COLS if rama == "woe" else art_pre["columnas_woe"]),
    n_filas_train=len(X_todo),
    tasa_default=float(y_todo.mean()),
    semilla=art_pre["metadatos"]["semilla"],
)
ruta_pre = artifacts.guardar(
    art_final, AQUI / "artifacts" / "preprocessors" / "preprocessor_final.joblib")
print(f"  Preprocesador emparejado guardado: {ruta_pre.name} "
      f"({len(X_todo):,} filas)")
if rama == "trees":
    print("  (rama 'trees': la limpieza es por reglas, no aprende del numero")
    print("   de filas; las tablas WoE se conservan solo por trazabilidad)")

# Comprobacion de cordura: sobre validacion (que ahora forma parte del train)
# el AUC tiene que ser >= al de antes. No es una medida de rendimiento real.
Xv = val[rama].drop(columns=[TARGET])[COLS] if rama == "trees" else X_todo.loc[val[rama].index]
print(f"  AUC in-sample sobre las filas de validacion: "
      f"{roc_auc_score(y_val, modelo_final.predict_proba(Xv)[:, 1]):.5f}")
print("  (in-sample: NO es una estimacion de rendimiento, solo cordura)")


# =============================================================================
# PASO 6: PREDICCION SOBRE cs-test.csv
# =============================================================================
titulo("PASO 6: SUBMISSION")

p_test = modelo_final.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"Id": np.arange(1, len(p_test) + 1), "Probability": p_test})
ruta = SUBMISSIONS / f"submission_{ELEGIDO}.csv"
sub.to_csv(ruta, index=False)

muestra = pd.read_csv(DATA / "sampleEntry.csv")
print(f"  Formato esperado (sampleEntry.csv): {list(muestra.columns)}, {len(muestra):,} filas")
print(f"  Generado                          : {list(sub.columns)}, {len(sub):,} filas")
print(f"\n{sub.head().to_string(index=False)}")

print("\n  --- VERIFICACIONES ---")
ok = True
def check(c, m):
    global ok
    print(f"  [{'OK' if c else 'ERROR'}] {m}")
    ok &= bool(c)

# El contrato del submission esta en src/evaluation.py y tiene tests propios.
for descripcion, correcto in validar_submission(sub, len(muestra)):
    check(correcto, descripcion)

print(f"\n  PD media predicha en test : {p_test.mean()*100:.2f}%")
print(f"  PD media real en train     : {y_todo.mean()*100:.2f}%")
dif = abs(p_test.mean() - y_todo.mean()) * 100
check(dif < 1.0, f"la PD media difiere solo {dif:.2f} pp del train "
                 "(train y test vienen de la misma poblacion)")
print(f"\n  Guardado: {ruta.relative_to(AQUI)}")

# Submissions de los otros modelos, para poder compararlos en Kaggle
for stem, a in modelos.items():
    if stem == ELEGIDO:
        continue
    if RAMA[stem] != "trees":
        continue
    Xt = pd.read_parquet(PROC / "trees" / "test.parquet")[a["columnas"]]
    pt = a["modelo"].predict_proba(Xt)[:, 1]
    pd.DataFrame({"Id": np.arange(1, len(pt) + 1), "Probability": pt}).to_csv(
        SUBMISSIONS / f"submission_{stem}.csv", index=False)
    print(f"  Guardado tambien: submission_{stem}.csv")


# =============================================================================
# PASO 7: GRAFICAS FINALES
# =============================================================================
titulo("PASO 7: GRAFICAS")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colores = {"logistic_v1": "#3a4f7a", "xgboost_v1": "#7a3a6a",
           "lightgbm_v1": "#1f7a5a", "ensemble": "#b08428"}

ax = axes[0, 0]
for stem, p in pred.items():
    f, t, _ = roc_curve(y_val, p)
    ax.plot(f, t, lw=2 if stem == ELEGIDO else 1.3, color=colores[stem], alpha=0.7,
            label=f"{NOMBRE[stem]} ({roc_auc_score(y_val, p):.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curvas ROC"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[0, 1]
for stem, p in pred.items():
    pr, rc, _ = precision_recall_curve(y_val, p)
    ax.plot(rc, pr, lw=2 if stem == ELEGIDO else 1.3, color=colores[stem],
            alpha=1.0 if stem == ELEGIDO else 0.75, label=NOMBRE[stem])
ax.axhline(y_val.mean(), ls="--", color="gray", lw=1, label="azar")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Curvas Precision-Recall"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
for stem, p in pred.items():
    d = pd.DataFrame({"p": p, "y": y_val.to_numpy()})
    d["dec"] = pd.qcut(d.p, 10, labels=False)
    g = d.groupby("dec").agg(pp_=("p", "mean"), pr=("y", "mean")) * 100
    ax.plot(g.pp_, g.pr, "o-", color=colores[stem], lw=1.6, ms=4, alpha=0.7,
            label=NOMBRE[stem])
lim = 40
ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="calibracion perfecta")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("PD predicha (%)"); ax.set_ylabel("PD observada (%)")
ax.set_title("Calibracion: cuanto mas pegado a la diagonal, mejor")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.hist(p_test, bins=80, color="#1f7a5a", alpha=0.85)
ax.axvline(UMBRAL_NEGOCIO, color="#a8392f", ls="--", lw=2,
           label=f"umbral de negocio {UMBRAL_NEGOCIO:.4f}")
ax.set_yscale("log")
ax.set_xlabel("PD predicha"); ax.set_ylabel("Personas (escala log)")
ax.set_title(f"Distribucion de las {len(p_test):,} predicciones de test")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
rechazados = (p_test >= UMBRAL_NEGOCIO).sum()
ax.text(0.55, 0.75, f"se rechazaria a\n{rechazados:,} ({rechazados/len(p_test)*100:.1f}%)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#a8392f"))

plt.tight_layout()
plt.savefig(PLOTS / "06_comparacion_final.png", dpi=110)
plt.close()
print(f"  {(PLOTS / '06_comparacion_final.png').relative_to(AQUI)}")

comp.to_csv(METRICAS / "comparacion_final.csv", index=False)
print(f"  artifacts/metrics/comparacion_final.csv")

print("\n" + "=" * 79)
print(f"MODELO FINAL: {elegido_nombre}   |   submission con {len(sub):,} predicciones"
      if ok else "HAY ERRORES: REVISAR ARRIBA")
print("=" * 79)
sys.exit(0 if ok else 1)
