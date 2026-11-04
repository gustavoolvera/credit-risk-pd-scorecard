# -*- coding: utf-8 -*-
"""
shap_xgboost.py: SHAP de XGBoost (analisis historico)
Proyecto: Give Me Some Credit: Probability of Default

Hasta el 2026-11-03 XGBoost era el modelo elegido y 06_evaluation.py calculaba
su SHAP. Desde la seleccion por CV en train (docs/04 §4.7.2), 06 explica
LightGBM. Este script regenera el SHAP de XGBoost con el mismo procedimiento
que usaba 06, para que las cifras citadas en E9 y en docs/02 §2.9 sigan siendo
reproducibles. Solo escribe en resultados/.

Se ejecuta con:  python experiments/shap_xgboost/shap_xgboost.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
SALIDA = AQUI / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

art = joblib.load(RAIZ / "artifacts" / "models" / "xgboost_v1.joblib")
val = pd.read_parquet(RAIZ / "datasets" / "processed" / "trees" / "val.parquet")
Xs = val.drop(columns=["SeriousDlqin2yrs"])[art["columnas"]]

explainer = shap.TreeExplainer(art["modelo"])
sv = explainer.shap_values(Xs.sample(3000, random_state=42))   # misma muestra que 06
imp = pd.DataFrame({"variable": art["columnas"], "impacto_medio": np.abs(sv).mean(axis=0)}) \
    .sort_values(["impacto_medio", "variable"], ascending=[False, True]).reset_index(drop=True)
imp["pct"] = (imp["impacto_medio"] / imp["impacto_medio"].sum() * 100).round(2)
gain = pd.Series(art["modelo"].feature_importances_, index=art["columnas"])
imp["gain_pct"] = imp["variable"].map(gain / gain.sum() * 100).round(2)
imp.to_csv(SALIDA / "shap_importancia_xgboost.csv", index=False)
print(imp.head(10).to_string(index=False))

p = art["modelo"].predict_proba(Xs)[:, 1]
i = int(np.argmax(p))
c = explainer.shap_values(Xs.iloc[[i]])[0]
det = pd.DataFrame({"variable": art["columnas"], "valor": Xs.iloc[i].values, "contribucion": c})
det = det.reindex(det["contribucion"].abs().sort_values(ascending=False).index)
det.to_csv(SALIDA / "explicacion_cliente.csv", index=False)
base = float(explainer.expected_value)
print(f"\nCliente #{Xs.index[i]}  PD = {p[i] * 100:.1f}%  base {base:+.4f}  "
      f"suma {base + c.sum():+.4f}")
print(det.head(5).to_string(index=False))
