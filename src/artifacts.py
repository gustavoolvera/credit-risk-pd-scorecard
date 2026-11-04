# -*- coding: utf-8 -*-
"""
artifacts.py: Persistencia de los transformadores aprendidos
Proyecto: Give Me Some Credit: Probability of Default

POR QUE ESTE MODULO EXISTE

Guardar solo los .parquet no basta. Un dataset procesado es el RESULTADO
de una transformacion, pero no la transformacion en si.

Si manana llega un cliente nuevo y hay que puntuarlo, necesitas aplicarle
EXACTAMENTE los mismos cortes de bins y los mismos valores WoE que se
aprendieron el dia del entrenamiento. Sin eso tendrias que re-ejecutar todo
el pipeline, y bastaria que cambiara una constante o la semilla para que el
scorecard dejara de ser el mismo: y ya no podrias explicar por que una
persona recibio la puntuacion que recibio.

En riesgo de modelo esto es un requisito, no una comodidad: el scorecard
tiene que ser un objeto CONGELADO, versionado y auditable.

Lo que se guarda:
  - las tablas WoE aprendidas (los valores por bin)
  - los cortes de los bins
  - los umbrales de limpieza vigentes al entrenar
  - el orden y nombre exacto de las columnas de salida
  - metadatos: fecha, semilla, filas usadas, tasa de default
"""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from . import preprocessing as pp
from . import woe as woe_mod


def construir_artefacto(tablas_woe, columnas_trees, columnas_woe,
                        n_filas_train, tasa_default, semilla):
    """
    Empaqueta todo lo necesario para reproducir la transformacion
    sobre datos nuevos, sin volver a entrenar.
    """
    return {
        "version": "1.0",
        "creado": datetime.now().isoformat(timespec="seconds"),

        # --- lo aprendido de los datos (depende del train) ---
        "tablas_woe": tablas_woe,

        # --- la configuracion usada (no depende de los datos, pero debe
        #     quedar registrada: si cambia, el artefacto ya no es el mismo) ---
        "config_limpieza": {
            "SENTINELAS": pp.SENTINELAS,
            "REVUTIL_UMBRAL_INVALIDO": pp.REVUTIL_UMBRAL_INVALIDO,
            # DebtRatio no tiene umbral: se separa por existencia de
            # denominador (ver paso 1d de preprocessing.limpiar).
            "DEBTRATIO_REGLA": "sin_denominador = income_faltante | income_cero",
            "EDAD_MINIMA": pp.EDAD_MINIMA,
            "INGRESO_MINIMO_PLAUSIBLE": pp.INGRESO_MINIMO_PLAUSIBLE,
        },
        "bins": woe_mod.BINS_TODOS,

        # --- contrato de salida: si el orden cambia, el modelo se rompe ---
        "columnas_trees": list(columnas_trees),
        "columnas_woe": list(columnas_woe),

        # --- trazabilidad ---
        "metadatos": {
            "n_filas_train": int(n_filas_train),
            "tasa_default_train": float(tasa_default),
            "semilla": int(semilla),
        },
    }


def guardar(artefacto, ruta):
    """Guarda el artefacto en disco."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ruta)
    return ruta


def cargar(ruta):
    """Carga un artefacto previamente guardado."""
    return joblib.load(Path(ruta))


def transformar_datos_nuevos(df_crudo, artefacto, rama="woe"):
    """
    Aplica a datos NUEVOS exactamente la misma transformacion que se uso
    al entrenar. Esta es la funcion que usarias en produccion.

    No mira el target ni aprende nada: solo aplica lo que ya estaba guardado.

    rama: "woe" (para regresion logistica) o "trees" (para XGBoost/LightGBM)
    """
    df = pp.crear_features(pp.limpiar(df_crudo))

    if rama == "trees":
        salida = pp.preparar_arboles(df)
        columnas = artefacto["columnas_trees"]
    elif rama == "woe":
        salida = woe_mod.aplicar_woe(df, artefacto["tablas_woe"])
        columnas = artefacto["columnas_woe"]
    else:
        raise ValueError(f"rama debe ser 'woe' o 'trees', no {rama!r}")

    # Reordenamos al contrato guardado. Si falta una columna es un error real:
    # significa que el codigo cambio respecto a cuando se entreno el modelo.
    faltantes = [c for c in columnas if c not in salida.columns]
    if faltantes:
        raise ValueError(
            f"El artefacto espera columnas que el codigo actual no genera: "
            f"{faltantes}. El pipeline cambio desde que se entreno el modelo."
        )

    return salida[columnas]


def resumen(artefacto):
    """Devuelve un resumen legible del artefacto, para inspeccionarlo."""
    m = artefacto["metadatos"]
    lineas = [
        f"Artefacto version {artefacto['version']}: creado {artefacto['creado']}",
        f"  Entrenado con : {m['n_filas_train']:,} filas",
        f"  Default rate  : {m['tasa_default_train']*100:.2f}%",
        f"  Semilla       : {m['semilla']}",
        f"  Variables WoE : {len(artefacto['tablas_woe'])}",
        f"  Columnas trees: {len(artefacto['columnas_trees'])}",
        f"  Columnas woe  : {len(artefacto['columnas_woe'])}",
        "  Umbrales de limpieza:",
    ]
    for k, v in artefacto["config_limpieza"].items():
        lineas.append(f"    {k:<28} = {v}")
    return "\n".join(lineas)


# =============================================================================
# RUTAS Y RESULTADOS GUARDADOS
# =============================================================================

def ruta_datos(raiz):
    """
    Carpeta con los CSV de Kaggle (cs-training.csv, cs-test.csv, sampleEntry.csv).

    Orden de busqueda: variable de entorno PD_DATA_DIR, data/ dentro del
    repositorio y, por compatibilidad con la estructura original del autor,
    ../../data respecto al repositorio.
    """
    raiz = Path(raiz).resolve()
    candidatas = ([Path(os.environ["PD_DATA_DIR"])] if os.environ.get("PD_DATA_DIR") else [])         + [raiz / "data", raiz.parent.parent / "data"]
    for c in candidatas:
        if (c / "cs-training.csv").exists():
            return c
    raise FileNotFoundError(
        "No se encuentra cs-training.csv. Descarga los datos de "
        "https://www.kaggle.com/c/GiveMeSomeCredit/data y colocalos en "
        f"{raiz / 'data'} (o define PD_DATA_DIR). Buscado en: "
        + ", ".join(str(c) for c in candidatas))


def cargar_json(ruta):
    """Lee un JSON de resultados; devuelve {} si todavia no existe."""
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def guardar_json(datos, ruta):
    """Guarda un diccionario de resultados como JSON legible."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=float),
                    encoding="utf-8")
    return ruta
