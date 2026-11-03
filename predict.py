# -*- coding: utf-8 -*-
"""
predict.py — Puntuar solicitantes nuevos
Proyecto: Give Me Some Credit — Probability of Default

Inferencia sobre solicitantes nuevos. No entrena nada: carga los artefactos
guardados y los aplica.

    entrada: CSV con solicitantes (mismas columnas que cs-test.csv)
    salida : CSV con PD, score y decision

Uso:
    python predict.py --entrada solicitantes.csv
    python predict.py --entrada solicitantes.csv --modelo logistic_v1
    python predict.py --entrada solicitantes.csv --margen 800 --perdida 6000
    python predict.py --demo          # ejemplo con tres perfiles inventados

Puntua con los artefactos congelados (modelo y preprocesador emparejados),
asi que la PD no depende de volver a ejecutar el pipeline.

Aqui se cargan dos ficheros y se aplica lo que ya estaba decidido.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
from src import artifacts
from src.evaluation import umbral_optimo

MODELOS = AQUI / "artifacts" / "models"
PRE_DIR = AQUI / "artifacts" / "preprocessors"


def resolver_artefactos(modelo):
    """
    Devuelve (ruta_modelo, ruta_preprocesador) emparejados.

    'final' -> el modelo que eligio 06, reentrenado con 150,000 filas, con el
               preprocesador que 06 congelo junto a el.
    '<x>_v1' -> el modelo entrenado con train (120,000) en 03/04/05, con el
               preprocesador de 02.
    Mezclar un modelo con el preprocesador de otra "epoca" daria PD que nadie
    podria reproducir despues.
    """
    if modelo == "final":
        elegido = artifacts.cargar_json(AQUI / "artifacts" / "metrics" / "modelo_elegido.json")
        if not elegido:
            sys.exit("No existe modelo_elegido.json. Ejecuta primero 06_evaluation.py.")
        return (MODELOS / "final" / f"{elegido['modelo']}_final.joblib",
                PRE_DIR / "preprocessor_final.joblib")
    return MODELOS / f"{modelo}.joblib", PRE_DIR / "preprocessor_v1.joblib"

TARGET = "SeriousDlqin2yrs"

# Escala del scorecard — debe coincidir con 08_scorecard.py
SCORE_BASE, ODDS_BASE, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = SCORE_BASE - FACTOR * np.log(ODDS_BASE)

COLUMNAS_REQUERIDAS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def score_desde_pd(p):
    """Convierte probabilidad de mora en puntos de scorecard."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return OFFSET + FACTOR * np.log((1 - p) / p)


def datos_demo():
    """Tres perfiles inventados para ver el script en accion."""
    return pd.DataFrame([
        # Perfil solido: poco credito usado, sin atrasos, ingreso alto
        dict(RevolvingUtilizationOfUnsecuredLines=0.08, age=52,
             **{"NumberOfTime30-59DaysPastDueNotWorse": 0}, DebtRatio=0.28,
             MonthlyIncome=8500.0, NumberOfOpenCreditLinesAndLoans=7,
             NumberOfTimes90DaysLate=0, NumberRealEstateLoansOrLines=1,
             **{"NumberOfTime60-89DaysPastDueNotWorse": 0}, NumberOfDependents=1.0),
        # Perfil intermedio: credito medio usado, un atraso leve
        dict(RevolvingUtilizationOfUnsecuredLines=0.62, age=38,
             **{"NumberOfTime30-59DaysPastDueNotWorse": 1}, DebtRatio=0.47,
             MonthlyIncome=4200.0, NumberOfOpenCreditLinesAndLoans=9,
             NumberOfTimes90DaysLate=0, NumberRealEstateLoansOrLines=1,
             **{"NumberOfTime60-89DaysPastDueNotWorse": 0}, NumberOfDependents=2.0),
        # Perfil de riesgo: tarjeta al limite, historial de 90 dias
        dict(RevolvingUtilizationOfUnsecuredLines=0.97, age=29,
             **{"NumberOfTime30-59DaysPastDueNotWorse": 2}, DebtRatio=0.71,
             MonthlyIncome=2100.0, NumberOfOpenCreditLinesAndLoans=5,
             NumberOfTimes90DaysLate=1, NumberRealEstateLoansOrLines=0,
             **{"NumberOfTime60-89DaysPastDueNotWorse": 1}, NumberOfDependents=3.0),
    ], index=pd.Index([1, 2, 3], name="row_id"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, help="CSV con solicitantes")
    ap.add_argument("--salida", type=Path, default=None, help="CSV de resultados")
    ap.add_argument("--modelo", default="final",
                    help="final (el elegido en 06, por defecto) | xgboost_v1 | "
                         "lightgbm_v1 | logistic_v1")
    ap.add_argument("--margen", type=float, default=800,
                    help="ganancia si el cliente paga")
    ap.add_argument("--perdida", type=float, default=6000,
                    help="perdida si el cliente cae en mora")
    ap.add_argument("--demo", action="store_true",
                    help="usa tres perfiles de ejemplo en vez de un CSV")
    args = ap.parse_args()

    if not args.demo and args.entrada is None:
        ap.error("indica --entrada RUTA.csv, o usa --demo")

    # --- 1. Cargar los artefactos --------------------------------------------
    f_modelo, f_pre = resolver_artefactos(args.modelo)
    if not f_modelo.exists():
        sys.exit(f"No existe {f_modelo}. Ejecuta el pipeline (03-06) primero.")
    if not f_pre.exists():
        sys.exit(f"No existe {f_pre}. Ejecuta 02 (y 06 para el modelo final).")

    art_pre = artifacts.cargar(f_pre)
    art_mod = joblib.load(f_modelo)
    rama = art_mod.get("rama", "woe" if "logistic" in f_modelo.stem else "trees")

    print(f"Preprocesador : {f_pre.name}  (v{art_pre['version']}, "
          f"{art_pre['creado'][:10]})")
    print(f"Modelo        : {f_modelo.name}  ({len(art_mod['columnas'])} variables, "
          f"rama '{rama}')")

    # --- 2. Leer los solicitantes --------------------------------------------
    if args.demo:
        crudo = datos_demo()
        print(f"Entrada       : 3 perfiles de demostracion")
    else:
        crudo = pd.read_csv(args.entrada, index_col=0)
        crudo.index.name = "row_id"
        print(f"Entrada       : {args.entrada.name}  ({len(crudo):,} solicitantes)")

    crudo = crudo.drop(columns=[TARGET], errors="ignore")

    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in crudo.columns]
    if faltan:
        sys.exit(f"Al CSV le faltan columnas obligatorias: {faltan}")

    # --- 3. Transformar y puntuar --------------------------------------------
    # transformar_datos_nuevos aplica EXACTAMENTE las mismas reglas y tablas
    # WoE del dia del entrenamiento. No aprende nada del CSV de entrada.
    X = artifacts.transformar_datos_nuevos(crudo, art_pre, rama=rama)
    pd_pred = art_mod["modelo"].predict_proba(X[art_mod["columnas"]])[:, 1]

    umbral = umbral_optimo(args.margen, args.perdida)
    score = score_desde_pd(pd_pred)

    res = pd.DataFrame({
        "PD": np.round(pd_pred, 6),
        "PD_pct": np.round(pd_pred * 100, 2),
        "score": np.round(score, 0).astype(int),
        "decision": np.where(pd_pred < umbral, "APROBAR", "RECHAZAR"),
    }, index=crudo.index)

    # --- 4. Reportar ----------------------------------------------------------
    print(f"\nRegla de negocio: aprobar si PD < margen/(margen+perdida) = "
          f"{umbral*100:.2f}%")
    print(f"  margen ${args.margen:,.0f} | perdida ${args.perdida:,.0f} "
          f"| corte {score_desde_pd(umbral):.0f} puntos")

    print(f"\n{'-'*66}")
    if len(res) <= 20:
        print(res.to_string())
    else:
        print(res.head(10).to_string())
        print(f"  ... ({len(res)-10:,} filas mas)")
    print(f"{'-'*66}")

    n_ap = int((res.decision == "APROBAR").sum())
    print(f"\nAprobados : {n_ap:>7,}  ({n_ap/len(res)*100:5.1f}%)")
    print(f"Rechazados: {len(res)-n_ap:>7,}  ({(len(res)-n_ap)/len(res)*100:5.1f}%)")
    print(f"PD media  : {pd_pred.mean()*100:.2f}%")

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.salida)
        print(f"\nGuardado en {args.salida}")


if __name__ == "__main__":
    main()
