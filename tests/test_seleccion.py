# -*- coding: utf-8 -*-
"""
test_seleccion.py: Regla de eleccion del modelo final

Se ejecuta con:  pytest tests/ -v

La regla decide que modelo se despliega. Un error aqui cambiaria el modelo de
produccion sin que ninguna metrica lo delatara.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.seleccion import regla_seleccion, MARGEN_AUC


def medias(**modelos):
    return pd.DataFrame(modelos, index=["AUC", "Calib_pp"]).T


def test_dentro_del_margen_desempata_la_calibracion():
    """Un AUC algo mayor no gana si el otro modelo esta dentro del margen."""
    m = medias(xgb=(0.8653, 0.35), lgbm=(0.8652, 0.32), log=(0.8570, 0.64))
    elegido, elegibles = regla_seleccion(m)
    assert elegido == "lgbm"
    assert set(elegibles) == {"xgb", "lgbm"}


def test_fuera_del_margen_no_compite_por_calibracion():
    """Un modelo mejor calibrado pero con AUC fuera del margen queda excluido."""
    m = medias(a=(0.8700, 0.50), b=(0.8600, 0.10))
    elegido, elegibles = regla_seleccion(m)
    assert elegido == "a"
    assert elegibles == ["a"]


def test_el_limite_del_margen_es_inclusivo():
    m = medias(a=(0.8700, 0.50), b=(0.8700 - MARGEN_AUC, 0.10))
    assert regla_seleccion(m)[0] == "b"


def test_exige_las_columnas_necesarias():
    with pytest.raises(ValueError):
        regla_seleccion(pd.DataFrame({"AUC": [0.86]}, index=["a"]))
