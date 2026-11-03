# -*- coding: utf-8 -*-
"""
07_business_evaluation.py — Traduccion del modelo a dolares
Proyecto: Give Me Some Credit — Probability of Default

POR QUE ESTE SCRIPT EXISTE

Traduce las PD a beneficio esperado: punto de corte optimo M/(M+L), valor de
cada modelo frente a no usar ninguno y sensibilidad al ratio perdida/margen.
Un AUC alto no garantiza valor economico: depende de ese ratio.

Se ejecuta con:  python 07_business_evaluation.py
Evalua los modelos *_v1 de artifacts/models/; la sensibilidad usa el elegido en 06.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # signo "-" normal en los ejes

sys.path.insert(0, str(Path(__file__).parent))
from src import artifacts
from src.evaluation import umbral_optimo, beneficio, beneficio_sin_modelo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
for d in (METRICAS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"

# =============================================================================
# SUPUESTOS ECONOMICOS
# =============================================================================
# Simplificados a proposito: todos los prestamos del mismo importe, sin valor
# temporal del dinero ni recuperaciones parciales. La ESTRUCTURA del
# razonamiento es la que se usa en la practica; los numeros exactos los
# aportaria el area de negocio.
PRESTAMO = 10_000     # importe medio
MARGEN = 800          # ganancia si el cliente paga (8% sobre el principal)
LGD = 0.60            # Loss Given Default: fraccion del principal que se pierde
PERDIDA = PRESTAMO * LGD


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — DATOS Y MODELOS")

modelos = {}
# Solo los modelos entrenados con train (*_v1). El modelo final de 06 vive en
# models/final/ y ya vio las filas de validacion.
for f in sorted(MODELOS.glob("*_v1.joblib")):
    art = joblib.load(f)
    rama = "woe" if "logistic" in f.stem else "trees"
    modelos[f.stem] = {"art": art, "rama": rama}
    print(f"  {f.stem:<22} rama={rama}  ({len(art['columnas'])} variables)")

if not modelos:
    sys.exit("No hay modelos en artifacts/models/. Ejecuta antes 03_train_logistic.py")

datos = {}
for rama in {m["rama"] for m in modelos.values()}:
    va = pd.read_parquet(PROC / rama / "val.parquet")
    datos[rama] = (va.drop(columns=[TARGET]), va[TARGET].to_numpy())

y = datos[next(iter(datos))][1]
print(f"\n  Validacion: {len(y):,} personas | {int(y.sum()):,} morosos ({y.mean()*100:.2f}%)")

predicciones = {}
for nombre, m in modelos.items():
    X, _ = datos[m["rama"]]
    predicciones[nombre] = m["art"]["modelo"].predict_proba(X[m["art"]["columnas"]])[:, 1]


# =============================================================================
# PASO 2 — SUPUESTOS Y UMBRAL OPTIMO
# =============================================================================
titulo("PASO 2 — SUPUESTOS ECONOMICOS")

u_opt = umbral_optimo(MARGEN, PERDIDA)
print(f"  Prestamo medio           : ${PRESTAMO:,}")
print(f"  Ganancia si paga         : ${MARGEN:,}")
print(f"  Perdida si cae en mora   : ${PERDIDA:,.0f}   (LGD {LGD*100:.0f}%)")
print(f"  Ratio perdida / ganancia : {PERDIDA/MARGEN:.1f}x")
print(f"\n  UMBRAL OPTIMO = M / (M + L) = {MARGEN} / {MARGEN+PERDIDA:.0f} = {u_opt:.4f}"
      f"  ({u_opt*100:.2f}%)")
print("  -> aprobar a todo el que tenga PD por debajo de ese valor")


# =============================================================================
# PASO 3 — BENEFICIO POR UMBRAL
# =============================================================================
titulo("PASO 3 — BENEFICIO SEGUN EL UMBRAL DE APROBACION")

# Politicas de referencia, sin modelo
ben_todos = beneficio_sin_modelo(y, MARGEN, PERDIDA)
print(f"  {'Aprobar a TODOS (sin modelo)':<38} ${ben_todos:>14,.0f}")
print(f"  {'Aprobar a NADIE':<38} ${0:>14,.0f}")

for nombre, p in predicciones.items():
    print(f"\n  --- {nombre} ---")
    print(f"  {'umbral':>8} | {'aprobados':>10} | {'buenos':>8} | {'morosos':>8} | {'beneficio':>15}")
    print("  " + "-" * 64)
    filas = []
    for u in [0.02, 0.04, 0.06, 0.08, u_opt, 0.15, 0.20, 0.30, 0.50]:
        r = beneficio(p, y, u, MARGEN, PERDIDA)
        filas.append(r)
        marca = "  <- optimo teorico" if abs(u - u_opt) < 1e-9 else ""
        print(f"  {u:>8.4f} | {r['aprobados']:>10,} | {r['buenos']:>8,} | "
              f"{r['morosos']:>8,} | ${r['beneficio']:>14,.0f}{marca}")

    # --- ¿Merece la pena buscar el umbral en los datos? --------------------
    # Un umbral optimizado sobre el mismo conjunto donde se reporta el
    # beneficio queda ajustado a esas filas. Se comprueba partiendo validacion
    # en dos mitades: optimo en A, aplicado a B.
    mitad = len(y) // 2
    pA, yA = p[:mitad], y[:mitad]
    pB, yB = p[mitad:], y[mitad:]

    rejilla = np.linspace(0.005, 0.6, 300)
    u_emp_A = max(rejilla, key=lambda u: beneficio(pA, yA, u, MARGEN, PERDIDA)["beneficio"])

    ben_emp_B = beneficio(pB, yB, u_emp_A, MARGEN, PERDIDA)["beneficio"]
    ben_teo_B = beneficio(pB, yB, u_opt, MARGEN, PERDIDA)["beneficio"]

    print(f"\n  ¿Conviene BUSCAR el umbral en los datos o usar la formula?")
    print(f"    Umbral empirico hallado en la mitad A : {u_emp_A:.4f}")
    print(f"    Umbral teorico M/(M+L)                : {u_opt:.4f}")
    print(f"    Aplicados a la mitad B (nunca vista):")
    print(f"      empirico -> ${ben_emp_B:>12,.0f}")
    print(f"      teorico  -> ${ben_teo_B:>12,.0f}   "
          f"{'(gana la formula)' if ben_teo_B >= ben_emp_B else '(gana el empirico)'}")
    print(f"    -> Diferencia en B: ${ben_teo_B - ben_emp_B:+,.0f}. Se usa la formula porque no")
    print("       depende de la muestra (docs/05_experimentos.md, E12).")

    # El beneficio que se reporta usa SIEMPRE el umbral teorico: no depende
    # de haber mirado ningun conjunto de evaluacion.
    valor = beneficio(p, y, u_opt, MARGEN, PERDIDA)["beneficio"] - ben_todos
    print(f"\n  VALOR DEL MODELO (al umbral teorico): ${valor:,.0f} "
          f"sobre {len(y):,} solicitudes")
    print(f"                    = ${valor/len(y):,.2f} por solicitud evaluada")


# =============================================================================
# PASO 4 — ¿CUANDO NO VALE LA PENA EL MODELO?
# =============================================================================
titulo("PASO 4 — SENSIBILIDAD: ¿CUANDO COMPENSA USAR EL MODELO?")

print("  Valor del modelo segun el ratio perdida/ganancia.\n")

# La sensibilidad se calcula con el modelo que eligio 06, no con el primero
# que aparezca en disco.
elegido = artifacts.cargar_json(METRICAS / "modelo_elegido.json").get("modelo")
if elegido not in predicciones:
    sys.exit("Falta artifacts/metrics/modelo_elegido.json. Ejecuta 06_evaluation.py antes.")
nombre_principal = elegido
p = predicciones[nombre_principal]
print(f"  Modelo: {nombre_principal} (el elegido en 06)\n")

print(f"  {'ratio L/M':>10} {'umbral opt':>11} {'sin modelo':>16} {'con modelo':>16} {'ganancia':>15}")
print("  " + "-" * 74)
sens = []
for ratio in [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20]:
    L = MARGEN * ratio
    u = umbral_optimo(MARGEN, L)
    sin_m = beneficio_sin_modelo(y, MARGEN, L)
    con_m = beneficio(p, y, u, MARGEN, L)["beneficio"]
    con_m = max(con_m, 0)     # siempre existe la opcion de no prestar a nadie
    sens.append({"ratio_L_M": ratio, "umbral_optimo": round(u, 4),
                 "sin_modelo": sin_m, "con_modelo": con_m,
                 "ganancia": con_m - sin_m})
    print(f"  {ratio:>10.1f} {u:>11.4f} ${sin_m:>15,.0f} ${con_m:>15,.0f} ${con_m-sin_m:>14,.0f}")

sens = pd.DataFrame(sens)
bajo, alto = sens.iloc[0], sens.iloc[-1]
print("\n  LECTURA:")
print(f"   1. Con perdidas baratas ({bajo.ratio_L_M:g}x) el modelo apenas aporta: "
      f"${bajo.ganancia:+,.0f} sobre {len(y):,} solicitudes.")
print("      Casi todos quedarian aprobados igual.")
print(f"   2. Con perdidas caras ({alto.ratio_L_M:g}x) aportaria ${alto.ganancia:+,.0f}; "
      "por encima de ~15x el")


# =============================================================================
# PASO 5 — GRAFICAS
# =============================================================================
titulo("PASO 5 — GRAFICAS")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
COLOR = {"logistic_v1": "#3a4f7a", "xgboost_v1": "#7a3a6a", "lightgbm_v1": "#1f7a5a"}
ETIQ = {"logistic_v1": "Regresión logística (WoE)", "xgboost_v1": "XGBoost", "lightgbm_v1": "LightGBM"}
n_val = len(y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

# --- Beneficio vs umbral ----------------------------------------------------
# u_opt = M/(M+L) es el umbral de equilibrio: aprobar a un cliente con esa PD
# tiene beneficio esperado cero. El maximo empirico de cada curva se marca solo
# como referencia; la politica sigue usando u_opt.
ax = axes[0]
us = np.linspace(0.005, 0.6, 300)
for nombre, pred in predicciones.items():
    bs = np.array([beneficio(pred, y, u, MARGEN, PERDIDA)["beneficio"] for u in us]) / 1e6
    etiqueta = ETIQ.get(nombre, nombre) + (" (elegido)" if nombre == nombre_principal else "")
    ax.plot(us * 100, bs, lw=2.2 if nombre == nombre_principal else 1.5,
            color=COLOR.get(nombre), alpha=0.85, label=etiqueta)
    k = int(bs.argmax())
    ax.plot(us[k] * 100, bs[k], marker="o", ms=5, mfc="white", mec=COLOR.get(nombre), mew=1.5)
ax.plot([], [], marker="o", ms=5, mfc="white", mec="gray", ls="none",
        label="máximo observado (referencia)")
ax.axhline(ben_todos / 1e6, ls="--", color="gray", lw=1.2,
           label=f"sin modelo: aprobar a todos (\\${ben_todos/1e6:.2f}M)")
ax.axvline(u_opt * 100, ls=":", color="black", lw=1.5,
           label=f"umbral de equilibrio M/(M+L) = {u_opt*100:.2f} %")
ax.set_xlim(0, 40)   # por encima de 40% las curvas solo descienden
ax.set_xlabel("Umbral de aprobación: PD máxima aceptada (%)")
ax.set_ylabel("Beneficio (millones de USD)")
# "\$" evita que matplotlib interprete el texto entre dos "$" como formula.
ax.set_title(f"Beneficio según el umbral — validación (n = {n_val:,})\n"
             f"margen \\${MARGEN} por préstamo pagado, pérdida \\${PERDIDA:,.0f} por default")
ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.grid(alpha=0.3)

# --- Sensibilidad al ratio ---------------------------------------------------
ax = axes[1]
x = np.arange(len(sens))
ancho = 0.38
ax.bar(x - ancho/2, sens["sin_modelo"] / 1e6, ancho, label="sin modelo (aprobar a todos)",
       color="#9a9a9a")
ax.bar(x + ancho/2, sens["con_modelo"] / 1e6, ancho, label=f"con {ETIQ.get(nombre_principal)}",
       color=COLOR.get(nombre_principal, "#2b6247"))
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{r:g}×" for r in sens["ratio_L_M"]])
ax.set_xlabel("Ratio pérdida / margen (L / M), con M = \\$800 fijo")
ax.set_ylabel("Beneficio (millones de USD)")
ax.set_title(f"¿Cuándo compensa el modelo? — {ETIQ.get(nombre_principal)}, validación\n"
             "umbral recalculado en cada escenario: M/(M+L)")
ax.legend(fontsize=8, loc="upper right", frameon=False)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS / "07_evaluacion_negocio.png", dpi=130)
plt.close()
print(f"  Guardado: {(PLOTS / '07_evaluacion_negocio.png').relative_to(AQUI)}")

sens.to_csv(METRICAS / "business_sensibilidad.csv", index=False)
print(f"  Guardado: artifacts/metrics/business_sensibilidad.csv")

print("\n" + "=" * 79)
print("EVALUACION DE NEGOCIO COMPLETADA")
print("=" * 79)
