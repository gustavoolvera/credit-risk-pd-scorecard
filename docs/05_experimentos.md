# 05 — Registro de experimentos

[← Modelado](04_modelado.md) · [Índice](README.md) · [Siguiente: Negocio y regulación →](06_negocio_y_regulacion.md)

---

Este capítulo documenta **todo experimento ejecutado durante el desarrollo**,
incluidos los que fracasaron y los que corrigieron decisiones previas. Se
incluyen deliberadamente los errores cometidos: en validación de modelos, el
historial de decisiones descartadas es tan informativo como el resultado final.

---

## E1 — Umbral de `RevolvingUtilizationOfUnsecuredLines`

**Objetivo.** Determinar a partir de qué valor la variable deja de representar
deuda real.

**Hipótesis inicial (errónea).** Al ser un ratio, cualquier valor $>1$ es
inválido; capear en 1.

**Método.** Tasa de evento por banda, con la tasa global (6.68%) como referencia.

**Resultado [MEDIDO].**

| Banda | n | Tasa de evento | Lectura |
|---|---|---|---|
| 1.0 – 1.2 | 1,797 | 38.23% | señal |
| 1.2 – 1.5 | 363 | 46.83% | señal |
| 1.5 – 2.0 | 188 | 45.21% | señal |
| 2 – 100 | 148 | 14–57% | señal (muestra pequeña) |
| 100 – 1,000 | 33 | **3.03%** | ruido |
| > 1,000 | 139 | **7.19%** | ruido |

**Sensibilidad del umbral [MEDIDO].**

| Umbral | Filas descartadas | Tasa de evento de las descartadas | Veredicto |
|---|---|---|---|
| **1** | 2,640 | **37.50%** | ❌ destruye la señal más fuerte |
| 2 | 371 | 14.56% | ❌ aún descarta señal |
| 5 | 254 | 7.87% | aceptable |
| 20 | 234 | 5.56% | aceptable |
| **100** | 223 | **4.93%** | ✅ solo ruido |

**Interpretación.** La banda 1.0–2.0 es la **más predictiva del conjunto**
(hasta 46.83%, siete veces la tasa global). Son sobregiros y líneas excedidas
reales. La hipótesis inicial habría invalidado 2,640 registros con 37.50% de
evento.

**Confirmación independiente.** El histograma log-log muestra bimodalidad con un
valle vacío entre $10^1$ y $10^2$; el umbral de 100 cae exactamente ahí.

**Lección.** Una intuición matemáticamente razonable ("un ratio no supera 1")
puede ser catastrófica si contradice la evidencia empírica.

---

## E2 — Códigos centinela 96/98

**Objetivo.** Determinar si los valores 96 y 98 son datos o códigos.

**Método.** Distribución por valor exacto; tasa de evento; coincidencia entre
columnas; efecto sobre las correlaciones.

**Resultado [MEDIDO].** 214 filas con 54.67% de evento (vs 6.60% del resto).
**Las tres columnas de morosidad tienen el mismo valor simultáneamente en las
269**. Su eliminación reduce las correlaciones entre las tres variables de
~0.99 a ~0.25.

**Interpretación.** Código de sistema, no dato. El grupo más predictivo del
conjunto.

**Tratamiento.** `96/98 → NaN` + `flag_registro_invalido`. Adicionalmente, los
valores legítimos se agrupan a partir de 4 porque la relación satura.

---

## E3 — Ingresos codificados como cero

**Objetivo.** Verificar si `MonthlyIncome = 0` es un ingreso real.

**Motivación.** Detectado al sustituir el histograma recortado al percentil 99
por uno de rango completo, que reveló un pico anómalo.

**Resultado [MEDIDO].** Ver [§3.2.3](03_pipeline_datos.md#323-monthlyincome--1-medido).

**Interpretación.** Mismo patrón que E1 y E2: código disfrazado de número, con
riesgo **inverso** al que el valor sugiere.

**Impacto medido tras el tratamiento.**

| Métrica | Antes | Después |
|---|---|---|
| IV de `MonthlyIncome` | 0.0703 | **0.0784** |
| IV de `ingreso_por_persona` | 0.0903 (débil) | **0.1049 (medio)** |

**Lección metodológica.** El recorte al percentil 99 en el EDA ocultaba tres
patologías. **Un histograma que se ve aplastado por un outlier está haciendo su
trabajo: esa fealdad es el hallazgo.**

---

## E4 — Bug en `peor_atraso`

**Objetivo.** (No planificado — detectado al revisar las tablas WoE.)

**Síntoma [MEDIDO].**

```
total_atrasos   bin (-0.1, 0.0]  →  n=95,692  evento 2.74%  + bin FALTANTE n=214 (54.67%)
peor_atraso     bin (-0.1, 0.0]  →  n=95,906  evento 2.86%  ← sin bin FALTANTE
```

Diferencia exacta: **214 filas**, las de los centinelas.

**Causa.** `peor_atraso` se calculaba con comparaciones `> 0`. Como `NaN > 0`
evalúa a `False`, las filas con centinela quedaban en `peor_atraso = 0` ("sin
atrasos") — el grupo más seguro — teniendo 54.67% de evento.

**Corrección.**

```python
sin_historial = df[COLS_MORA].isna().all(axis=1)
df.loc[sin_historial, "peor_atraso"] = np.nan
```

**Impacto [MEDIDO].** IV de `peor_atraso`: 1.3326 → **1.4052**. El bin limpio
bajó de 2.86% a 2.74%.

**Prevención.** Prueba de regresión
`test_regresion_peor_atraso_no_traga_centinelas`.

---

## E5 — Selección de variables para la regresión logística

**Objetivo.** Elegir el subconjunto que evite multicolinealidad.

**Resultado [MEDIDO].** Ver [§4.2.2](04_modelado.md#422-selección-de-variables).

**Hallazgo central.** El subconjunto con **mejor AUC** (B, 0.86300) tenía **4
coeficientes con signo invertido**, incluido uno que afirmaba "más episodios de
mora leve → menos riesgo".

**Causa raíz.** Todas las variables derivadas —no solo las de morosidad—
competían con sus variables padre:

| Derivada | Compite con |
|---|---|
| `ingreso_por_persona` | `MonthlyIncome`, `NumberOfDependents` |
| `prop_inmobiliaria` | `NumberRealEstateLoansOrLines`, `NumberOfOpenCreditLinesAndLoans` |
| `deuda_bruta_proxy` | `DebtRatio_limpio` |
| `total_atrasos`, `peor_atraso`, `flag_algun_atraso` | Las tres originales de mora |

**Decisión.** Cambiar el criterio de selección: **exigir signos coherentes
primero**, maximizar AUC después.

**Coste [MEDIDO].** +0.00076 de AUC (0.088%).

> **Nota posterior.** Este experimento solo revisaba las variables `woe_`. La
> auditoría descubrió después que la comprobación era incompleta y que tres
> indicadores llegaban al modelo con el signo contrario a su comportamiento
> univariado. Ver **E16**.

**Lección.** La ingeniería de variables puede restar. Siete variables con lógica
de negocio sólida no aportaron poder predictivo y rompieron la interpretabilidad.

---

## E6 — Sensibilidad al parámetro `C`

**Objetivo.** Determinar si la regularización importa.

**Resultado [MEDIDO].** Ver [§2.4.4](02_fundamentos_matematicos.md#244-regularización).

**Interpretación.** Con ~6,666 observaciones por parámetro, los datos dominan por
completo. El AUC de entrenamiento (0.85913) es **menor** que el de validación
(0.86272), lo que descarta sobreajuste y por tanto la necesidad de regularizar.

**Lección.** Saber qué **no** ajustar ahorra tanto tiempo como saber qué ajustar.
Los parámetros que movieron la aguja en este proyecto fueron los de limpieza
(umbrales, centinelas), no los del algoritmo.

---

## E7 — Sobreajuste de XGBoost con valores por defecto

**Objetivo.** Establecer la línea base antes de ajustar.

**Resultado [MEDIDO].** `AUC ajuste 0.92919` / `AUC validación 0.85592` — brecha
de 7.3 puntos, y **peor que la regresión logística**.

Tras el ajuste: `0.87651 / 0.86854`, brecha de 0.8 puntos.

**Interpretación.** Los valores por defecto de XGBoost son agresivos. Es el error
más común al usar la biblioteca.

---

## E8 — ¿Aportan las derivadas a los árboles?

**Objetivo.** Verificar si la conclusión de E5 se extiende a modelos no lineales.

**Resultado [MEDIDO]** (sobre el conjunto de *early stopping*):

| Subconjunto | n | AUC ajuste | AUC early-stop |
|---|---|---|---|
| Todas | 25 | 0.87853 | 0.86350 |
| **Originales + indicadores** | 18 | 0.87619 | **0.86405** |
| Solo originales | 10 | 0.87616 | 0.86342 |

**Interpretación.** Mismo veredicto que E5, por razón distinta: en la rama lineal
las derivadas **rompían** el modelo (signos invertidos); en la rama de árboles
solo **redistribuyen el crédito** en la tabla de importancia sin aportar.

**Decisión.** Se conservan las 25 en la rama de árboles: la diferencia está
dentro del ruido y no hay coeficientes que puedan invertirse.

---

## E9 — Importancia de variables: inestabilidad entre algoritmos

**Objetivo.** Verificar si `feature_importances_` es interpretable.

**Resultado [MEDIDO].**

| Variable | Gain XGBoost | Gain LightGBM | SHAP medio (XGBoost) |
|---|---|---|---|
| `peor_atraso` | 25.7% | 22.6% | 9.5% |
| `total_atrasos` | 6.2% | **26.5%** | 15.1% |
| `flag_algun_atraso` | **38.9%** | 1.7% | 6.9% |
| `RevolvingUtilization` | **4.3%** | **24.3%** | **30.5%** |

Gain de `xgboost_v1` y `lightgbm_v1`; SHAP de XGBoost generado por
`experiments/shap_xgboost/` (actualizado el 2026-11-03).

**Interpretación.** `RevolvingUtilization` recibe 4.3% del gain en un algoritmo
y 24.3% en otro **con datos idénticos**. Con variables redundantes, cada
algoritmo reparte el crédito a su manera.

**Verificación complementaria [MEDIDO].** Las tres derivadas de atraso suman el
70.8% del gain de XGBoost y pueden eliminarse sin deterioro significativo del
AUC (-0.0001, IC 95 % [-0.0007, +0.0004]; E19, escenario C). **Importancia
alta ≠ imprescindible.**

**Lección.** La importancia nativa es una propiedad **del modelo entrenado**, no
de los datos. Presentarla como "esto es lo que predice la mora" es presentar un
artefacto.

---

## E10 — Comparación de los tres modelos (histórico)

Primera comparación en CV de 5 folds idénticos, cuando la regresión logística
aún tenía 18 variables (antes de E16). Concluyó que ambos boosters superan a la
logística y que entre ellos no hay diferencia detectable. Su regla de decisión
($|\bar d| > 2\,\text{DE}/\sqrt{5}$) no corregía la dependencia entre folds.
**Versión vigente:** [04 §4.7.1](04_modelado.md#471-incertidumbre-de-la-comparación-diferencias-pareadas),
con los modelos actuales e IC de Nadeau–Bengio. Las conclusiones no cambiaron.

---

## E11 — Validación cruzada anidada

**Objetivo.** Verificar si el AUC reportado está inflado por selección.

**Sospecha.** Los hiperparámetros se eligieron con un subconjunto del
entrenamiento que después aparece en los folds de test.

**Método.** Mismos 5 folds externos que la CV de `04` y `05` (150,000 filas).
Dentro de cada fold se repite toda la selección de `04` solo con su parte de
entrenamiento: split 80/20, rejilla de 11 configuraciones con early stopping, y
se elige la de mejor AUC en el 20 %. Después se reentrena con el entrenamiento
completo del fold y se evalúa en su parte de prueba. Comparación fold a fold
contra la CV reportada, con IC 95 % de Nadeau–Bengio.

**Resultado [MEDIDO]** (`experiments/cv_anidada/cv_anidada.py`, ~3 min):

| Fold | CV reportada | CV anidada | Configuración elegida en el fold |
|---|---|---|---|
| 1 | 0.87095 | 0.87134 | depth 3, mcw 1, sub 1.0, col 1.0 (287 árboles) |
| 2 | 0.86495 | 0.86581 | depth 4, mcw 10, sub 0.9, col 0.9, λ 5 (377) |
| 3 | 0.86179 | 0.86132 | depth 4, mcw 5, sub 0.8, col 0.8 (199) |
| 4 | 0.86577 | 0.86618 | depth 5, mcw 20, sub 0.8, col 0.6 (161) = producción |
| 5 | 0.86796 | 0.86768 | depth 7, mcw 20, sub 0.8, col 0.6 (107) |
| **Media (DE `ddof=0`)** | **0.86628 (0.00306)** | **0.86647 (0.00323)** | |

Diferencia anidada - reportada: **+0.00018**, IC 95 % [-0.00084, +0.00120],
p = 0.65.

**Interpretación.** No se detecta optimismo por la selección de
hiperparámetros. El IC no permite afirmar que el sesgo sea exactamente cero,
pero sí acotarlo por debajo de ~0.001 de AUC. Cada fold elige una configuración
distinta y solo uno coincide con la de producción: el AUC es casi plano en la
rejilla, y por eso elegir dentro de ella no infla el resultado.

**Corrección (2026-11-03).** La versión anterior de este experimento reportaba
0.86661 ± 0.00337 sin ningún script en el repositorio que lo respaldara, y esas
cifras no se reproducen. Se sustituyen por las anteriores, generadas por un
script versionado. La conclusión cualitativa no cambia.

---

## E12 — ¿Debe buscarse el umbral en los datos?

**Objetivo.** Contrastar el umbral teórico $M/(M+L)$ contra el hallado
empíricamente.

**Método.** Partir validación en dos mitades; hallar el óptimo en A; aplicarlo
en B.

**Resultado [MEDIDO]** (`07_business_evaluation.py`, beneficio en la mitad B,
15,000 solicitudes):

| Modelo | Umbral empírico (mitad A) | Empírico | Fórmula (0.1176) | Gana |
|---|---|---|---|---|
| XGBoost | 0.1244 | $7,937,600 | $7,959,600 | fórmula (+$22,000) |
| LogReg | 0.0985 | $7,832,400 | $7,901,600 | fórmula (+$69,200) |
| LightGBM | 0.1184 | $7,958,000 | $7,952,800 | empírico (+$5,200) |

**Interpretación.** Buscar el umbral en una mitad no mejora de forma consistente
a la fórmula: gana la fórmula en 2 de 3 modelos, y la única excepción es de
$5,200 sobre $8M. Con una sola partición en dos mitades no se puede afirmar más
que eso. La fórmula se prefiere porque no depende de la muestra, no porque se
haya demostrado superior.

**Corrección (2026-11-03).** La versión anterior citaba $7,993,200 frente a
$8,006,000 y concluía que la fórmula siempre gana. Esas cifras no coinciden
con la ejecución actual.

---

## E13 — Ensemble

**Objetivo.** Verificar si promediar mejora.

**Resultado [MEDIDO].** AUC 0.86839, por debajo de XGBoost solo (0.86854), y
peor calibración (0.453 pp vs 0.338 pp).

**Interpretación.** Los tres modelos se equivocan en las mismas personas
(correlación 0.9928 entre XGBoost y LightGBM). Sin diversidad de errores, no hay
ganancia por promediado.

---

## E14 — ¿Es aplicable el PSI?

**Objetivo.** Evaluar si procede implementar monitoreo de estabilidad
poblacional.

**Resultado [MEDIDO].**

```
PSI máximo de las 15 variables : 0.00073
PSI medio                       : 0.00018
Umbral de alerta convencional   : 0.10        ← 137× superior
```

**Interpretación.** El PSI detecta deriva **temporal**. Aquí train y test son
particiones aleatorias de la misma población de 2011: estructuralmente no hay
deriva que medir.

**Decisión.** No implementarlo. Se documenta por qué no aplica, lo cual es más
informativo que publicar un 0.0007 sin significado.

**Lección.** Aplicar una técnica donde no hay nada que medir es relleno, no
rigor.

---

## E15 — Reproducibilidad desde cero

**Método.** Copia limpia del repositorio (solo archivos versionables), entorno
virtual nuevo con `requirements.txt`, datos en `data/`, y ejecución de `01`–`08`,
`predict.py --demo`, los tests y los experimentos.

**Resultado [MEDIDO]** (2026-11-03). Datasets procesados, métricas, submissions
y resultados de experimentos idénticos a los publicados. Las únicas diferencias
son de punto flotante (≤ 2e-6). Con una versión distinta de XGBoost (3.4.1
frente a 3.3.0) las métricas de XGBoost cambian en la 7.ª cifra decimal, sin
alterar ninguna conclusión; por eso las versiones están fijadas.

Una versión anterior de este experimento (42 pruebas, LogReg de 18 variables)
queda en el historial de git.

---

## E16 — Corrección de H1: coherencia completa de signos

**Objetivo.** Eliminar los coeficientes que contradicen la dirección univariada
de su variable.

**Motivación.** La auditoría (capítulo [07](07_auditoria.md#h1)) detectó que
`signos_invertidos()` eximía a los indicadores, y que tres de ellos tenían el
signo al revés. El peor, `flag_registro_invalido`, identifica al grupo con
**54.67%** de eventos frente al 6.60% de la población, y su coeficiente negativo
le asignaba **puntos positivos** en el scorecard.

**Causa raíz [MEDIDO].** Colinealidad perfecta con los bins `FALTANTE`:

| Indicador | Variable WoE | Filas | Coincidencia |
|---|---|---|---|
| `flag_registro_invalido` | `woe_NumberOfTimes90DaysLate` | 214 | **100%** |
| `flag_debtratio_invalido` | `woe_DebtRatio_limpio` | 24,718 | **100%** |
| `flag_revutil_invalido` | `woe_RevolvingUtilizationOfUnsecuredLines` | 172 | **100%** |

Cuando un valor se convierte en `NaN` ocurren dos cosas a la vez: WoE lo coloca
en `FALTANTE` **y** se enciende el indicador. Las mismas filas contadas dos
veces.

**Método.** Nueva función `signos_incoherentes()` que verifica variables `woe_`
contra la convención de signo **e indicadores contra su dirección univariada
medida en los datos**. Ocho subconjuntos candidatos evaluados con el criterio
completo.

**Resultado [MEDIDO].**

| Subconjunto | n | AUC selección | VIF máx | Incoherentes |
|---|---|---|---|---|
| A: todas | 25 | 0.86135 | 80.5 | 8 |
| E: originales + indicadores (anterior) | 18 | 0.86059 | 8.8 | 3 |
| G: originales + `flag_algun_atraso` | 11 | 0.86006 | 3.6 | 1 |
| **H: solo las 10 originales** | **10** | 0.85749 | **1.3** | **0** |

**Hallazgo central.** Solo el subconjunto de las 10 variables originales alcanza
coherencia total. Retirar los tres indicadores colineales **desplaza** el problema
a otros (`flag_dependents_faltante`, luego
`woe_NumberOfOpenCreditLinesAndLoans`) en lugar de resolverlo.

**Coste [MEDIDO].**

| Métrica | Antes | Después | Δ |
|---|---|---|---|
| AUC (CV) | 0.85948 | 0.85794 | -0.00187 |
| KS | 0.5760 | 0.5696 | -0.0064 |
| Calibración | 0.53 pp | 0.75 pp | peor |
| VIF máximo | 8.8 | **1.3** | mejor |
| Valor de negocio | $175.33 | $171.95 | -$3.38 |

**Interpretación.** La corrección **empeora todas las métricas de rendimiento**.
Se acepta porque un scorecard cuya tabla de puntos contradice la evidencia no
sirve para su único propósito. A cambio, los predictores quedan prácticamente
ortogonales (VIF 1.3) y los coeficientes, estables.

**Lección.** En un scorecard WoE, la codificación de las variables originales ya
contiene los faltantes (bin `FALTANTE`), los extremos (bins de los bordes) y la
no-linealidad (el binning). **Todo indicador de calidad de dato es redundante por
construcción.**

**Prevención.** Seis pruebas nuevas, entre ellas
`test_la_comprobacion_completa_es_mas_estricta_que_la_antigua`, que verifica que
`signos_incoherentes()` detecta un superconjunto estricto de lo que detectaba la
comprobación anterior.

---

## E17 — Ablación de variables del modelo H

**Objetivo.** Determinar si las cuatro variables con menor permutation
importance pueden eliminarse sin deteriorar discriminación, calibración,
estabilidad ni beneficio.

**Motivación.** En `03b_logistic_interpretabilidad.py` cuatro variables tienen
una caída de AUC por permutación ≤ 0.002: `NumberRealEstateLoansOrLines`
(0.0019), `NumberOfDependents` (0.0001), `MonthlyIncome` (0.0000) y
`NumberOfOpenCreditLinesAndLoans` (-0.0001). Una importancia por permutación
cercana a cero **no** demuestra irrelevancia: con variables correlacionadas, la
información puede estar repartida.

**Método.**

- Solo las 120,000 filas de entrenamiento. Validación se usa una vez, como
  confirmación que no decide nada. Kaggle no se toca.
- CV estratificada repetida 5×5 (25 particiones), **idénticas para todos los
  escenarios** → comparaciones pareadas.
- Tablas WoE re-aprendidas **dentro** de cada partición. Mismos
  hiperparámetros que producción (`tol=1e-8`, `C=1`).
- Incertidumbre: t-test con varianza corregida de Nadeau & Bengio (2003), porque
  las particiones de una CV repetida se solapan.
- Escenarios: A (10 variables), B1–B4 (quitar cada candidata), C (quitar las
  tres de menor importancia), D (quitar las cuatro) y E (quitar `MonthlyIncome`
  y `NumberOfOpenCreditLinesAndLoans`, las dos con importancia ≈ 0 en ambos
  métodos de §4.8.6).

**Relevancia individual [MEDIDO].**

| Variable | IV | AUC sola | β (25 particiones) | % particiones con β < 0 | ρ máx. |
|---|---|---|---|---|---|
| RealEstateLoans | 0.058 | 0.562 | -0.604 ± 0.025 | 100 % | 0.28 |
| Dependents | 0.036 | 0.550 | -0.256 ± 0.029 | 100 % | 0.28 |
| MonthlyIncome | 0.078 | 0.579 | -0.025 ± 0.025 | **80 %** | 0.33 (DebtRatio) |
| OpenCreditLines | 0.068 | 0.563 | -0.069 ± 0.022 | 100 % | 0.28 |

Las cuatro tienen información univariada. La pregunta es si aportan algo
**condicionado** a las demás.

**Resultado de la ablación [MEDIDO]** (diferencias contra A, IC 95 %):

| Escenario | n | ΔAUC | ΔAP | ΔLog Loss | ΔBeneficio ($/sol.) |
|---|---|---|---|---|---|
| B1: sin RealEstate | 9 | **-0.00177** [-0.0024, -0.0012] | -0.0006 | **+0.00042** | -0.88 |
| B2: sin Dependents | 9 | +0.00006 | **-0.0010** [-0.0016, -0.0004] | +0.00004 | +0.30 |
| B3: sin Income | 9 | +0.00002 | -0.00015 | -0.00001 | 0.00 |
| B4: sin OpenCredit | 9 | 0.00000 | +0.00028 | 0.00000 | -0.21 |
| C: sin Dep+Inc+Open | 7 | +0.00009 | **-0.00095** | +0.00003 | +0.15 |
| D: sin las 4 | 6 | **-0.00193** | -0.0010 | **+0.00048** | -0.40 |
| E: sin Inc+Open | 8 | +0.00001 | +0.00018 | -0.00001 | -0.32 [-0.94, +0.31] |

En negrita, las diferencias significativas sin corregir. La calibración no
cambia de forma significativa en ningún escenario.

**Revisión metodológica [MEDIDO].**

1. **Origen de los cortes.** Las tasas citadas en `src/woe.py` coinciden
   exactamente con los 120k de train (no con los 150k): los cortes se eligieron
   viendo el target de los mismos datos de la CV, sin ver validación. Repetida
   la CV con cortes ChiMerge derivados **dentro** de cada partición, el
   optimismo de A es de +0.00009 de AUC y E - A pasa a -0.00015: la conclusión
   no cambia.
2. **Multiplicidad.** Con Holm sobre las 56 pruebas, siguen siendo
   significativas 4 de 7, todas por quitar RealEstateLoans (B1 y D). La pérdida
   de AP de Dependents deja de serlo (p de Holm = 0.07).
3. **Ingreso.** `DebtRatio_limpio` es deuda/ingreso y su bin `FALTANTE`
   coincide fila a fila con ingreso faltante o cero (25,495 filas). Sin
   `MonthlyIncome`, el modelo E calibra igual por decil de ingreso: error medio
   0.315 pp frente a 0.301 pp, y la misma correlación de Spearman (0.976).
4. **No inferioridad de E frente a A.** Márgenes fijados antes de una réplica
   con particiones nuevas (semilla 20261101): AUC 0.001, AP 0.002, calibración
   +0.05 pp, beneficio $1/solicitud y log loss 0.0002. E es no inferior **y**
   equivalente en las cinco métricas, en las tres fuentes de evidencia (CV
   original, réplica y ChiMerge por partición).

**Sobreajuste.** La diferencia train–prueba de A ya es mínima (0.00055 de AUC,
0.00025 de log loss), con ~12,000 observaciones por parámetro. D la reduce a
0.00039 **perdiendo** 0.0019 de AUC fuera de muestra: es menos capacidad, no
menos sobreajuste. Quitar `MonthlyIncome` sí estabiliza a `DebtRatio_limpio`
(desviación de su β entre particiones: 0.0182 → 0.0134).

**Clasificación.**

| Variable | Clasificación | Motivo |
|---|---|---|
| RealEstateLoans | Mantener | Quitarla empeora AUC y log loss en las 25 particiones, y sobrevive a Holm. Su WoE en forma de U no lo captura ninguna otra variable |
| Dependents | Incierta | Pérdida de AP consistente pero pequeña, no significativa tras Holm y no confirmada en validación |
| MonthlyIncome | Candidata a eliminar | Redundante con DebtRatio. β con signo inestable |
| OpenCreditLines | Candidata a eliminar | Ninguna métrica empeora |

**Decisión: se mantiene H (10 variables).** E es *equivalente*, no *mejor*. Las
razones para no sustituir H son:

1. **No hay ganancia que justificar el cambio.** En validación (informativa) E
   iguala el AUC (0.86061), pierde $4,400 (-0.03 %) y su calibración es 0.025 pp
   peor. Todo cae dentro de los márgenes, pero nada mejora.
2. **Elegir E exige cambiar el criterio de selección después de ver los
   resultados.** Con el criterio vigente de 03 (máximo AUC entre los candidatos
   coherentes) sigue ganando H: AUC de selección de I - H = -0.00006 [IC90 -0.00015, +0.00002].
   Solo un criterio nuevo de parsimonia bajo no inferioridad elige E.
3. **Gobierno del ingreso.** Retirar `MonthlyIncome` de un scorecard puede
   chocar con requisitos de capacidad de pago que dependen de la jurisdicción y
   no se han confirmado con cumplimiento normativo.
4. **H ya cumple todos los controles**: 0 signos incoherentes y VIF 1.29.

**Seguimiento.** En cada recalibración se debe vigilar el signo de β de
`MonthlyIncome`: es negativo en producción (-0.0214), pero positivo en el 20 %
de las particiones. Si llega a invertirse, E es la alternativa ya validada.

**Lección.** Una importancia por permutación cercana a cero no basta para
eliminar una variable: `NumberRealEstateLoansOrLines` tenía 0.0019 y quitarla
cuesta lo mismo de forma significativa. Y equivalencia estadística no implica
que convenga cambiar: el modelo más simple solo se prefiere si el criterio de
parsimonia se fijó **antes** de mirar.

**Reproducibilidad.** `experiments/ablacion_variables/` contiene `ablacion.py`
(≈80 s), `revision_metodologica.py` (≈2 min) y los resultados en `resultados/`.
Si se adoptara E, el cambio consistiría en añadir a los candidatos de `03` el
subconjunto "H sin MonthlyIncome ni OpenCreditLines" y elegir el de menos
variables cuyo AUC de selección quede a menos de 0.001 del mejor. Ese parche se
redactó, se probó en una copia aislada (elegía E) y se descartó con la
decisión de mantener H.

---

## E18 — Variables de atraso en XGBoost: redundancia y riesgo de fuga

> Análisis de **XGBoost**, el modelo elegido hasta el 2026-11-03. Desde la selección
> por CV en train el elegido es LightGBM ([04 §4.7.2](04_modelado.md#472-uso-de-validación-en-la-elección-del-modelo));
> las conclusiones sobre redundancia del bloque de atrasos no dependen de esa elección.

**Objetivo.** Las variables con más importancia nativa en XGBoost
(`flag_algun_atraso`, `peor_atraso`, `NumberOfTimes90DaysLate`,
`total_atrasos`) describen el mismo comportamiento. Hay dos preguntas: si su
importancia es interpretable, y si se construyeron con información disponible
en el momento de la decisión.

**Definiciones verificadas [MEDIDO]** (`crear_features()`, en los 119,786
clientes de train con historial):

| Variable | Definición | Verificación |
|---|---|---|
| `total_atrasos` | Suma de los conteos de 30-59, 60-89 y 90+ días | Idéntica fila a fila |
| `flag_algun_atraso` | `total_atrasos > 0` | Idéntica fila a fila |
| `peor_atraso` | Peor nivel: 0 ninguno, 1 = 30-59, 2 = 60-89, 3 = 90+ | `peor_atraso = 3` ⇔ `NumberOfTimes90DaysLate > 0` |

Con centinela 96/98 las tres quedan como faltantes (historial desconocido).

**Redundancia [MEDIDO].** Las tres derivadas tienen correlación de Spearman de
0.99 entre sí, y de 0.48–0.56 con `NumberOfTimes90DaysLate`. En el modelo
`xgboost_v1` (validación, 10 permutaciones):

| Variable | Gain | Caída de AUC al permutar |
|---|---|---|
| `flag_algun_atraso` | **0.389** | 0.0025 |
| `peor_atraso` | 0.257 | 0.0045 |
| `NumberOfTimes90DaysLate` | 0.120 | 0.0014 |
| `total_atrasos` | 0.062 | 0.0156 |
| 30-59 / 60-89 días | 0.031 / 0.020 | 0.0010 / 0.0002 |
| **Suma individual de las 6** | | 0.025 |
| **Las 6 permutadas como bloque** | | **0.126** |
| `RevolvingUtilization` (referencia) | | 0.062 |

La importancia nativa y la permutación individual se contradicen: la primera
pone a `flag_algun_atraso` como la variable principal, y permutarla sola casi
no cambia el AUC. Medido como bloque, el historial de morosidad es la
información más importante del modelo, el doble que RevolvingUtilization.

**Fuga de información.**

- *Derivadas:* son funciones deterministas de cada fila, sin target ni
  estadísticas de otras filas. **No pueden introducir fuga nueva**; heredan el
  riesgo temporal de sus variables de origen.
- *Conteos originales:* según el diccionario de Kaggle, 30-59 y 60-89 cubren
  "los últimos 2 años" y el target cubre los 2 años **siguientes**.
  `NumberOfTimes90DaysLate` **no declara ventana**, y mide el mismo evento que
  el target. Prueba directa: si el conteo incluyera el periodo del target, todo
  moroso tendría al menos un evento previo. En train, el **65.2 % de los
  morosos tiene 0**, y tener un evento previo da 33.8 % de default (66.6 % con
  4 o más), no ~100 %. **La contaminación directa queda descartada.** Un
  solapamiento parcial (por ejemplo, una foto tomada meses después del inicio
  de la ventana) no se puede verificar sin fechas de observación, igual que la
  validación fuera de tiempo ([H2](07_auditoria.md#h2)). Es un supuesto que
  debe declararse.

**Lección.** Con variables redundantes la importancia debe reportarse
**agrupada por bloque**. Ni la importancia nativa ni la permutación individual
sirven para decir "esto es lo que predice la mora". La regresión logística de
producción no usa ninguna de las derivadas.

**Reproducibilidad.** Cálculo de solo lectura sobre `datasets/processed/trees/`
y `xgboost_v1.joblib`. La ablación del bloque con reentrenamiento está en E19.

---

## E19 — Ablación del bloque de atrasos en XGBoost

> Análisis de **XGBoost**, el modelo elegido hasta el 2026-11-03. Desde la selección
> por CV en train el elegido es LightGBM ([04 §4.7.2](04_modelado.md#472-uso-de-validación-en-la-elección-del-modelo));
> las conclusiones sobre redundancia del bloque de atrasos no dependen de esa elección.

**Objetivo.** E18 midió el bloque permutando un modelo ya entrenado. Aquí se
**reentrena** sin él, para separar dos cosas: la **redundancia interna** (si
las variables del bloque se sustituyen entre sí) y la **contribución
incremental** (si el resto de variables puede compensar el bloque completo).

**Método.**

- Bloque: 3 conteos originales (30-59, 60-89, 90+ días) y 3 derivadas
  (`total_atrasos`, `flag_algun_atraso`, `peor_atraso`).
- Escenarios: A (todas, 24), B (sin el bloque, 18), C (sin las derivadas, 21),
  D (sin los conteos originales, 21).
- CV estratificada repetida 5×5 sobre train, con las mismas particiones para
  los cuatro escenarios. En cada partición, el early stopping usa un 20 % de su
  parte de ajuste, igual que `04`.
- Hiperparámetros de producción constantes. Los eligió la rejilla de `04` con
  todas las variables, así que si acaso favorecen a A.
- IC 95 % con t-test corregido de Nadeau–Bengio. Validación se usa solo como
  confirmación. Control: A reentrenado reproduce el AUC de producción
  (0.86854).

**Resultado [MEDIDO]** (CV, diferencias contra A):

| Escenario | ΔAUC | ΔKS | ΔLog Loss | ΔAP | ΔBeneficio ($/sol.) |
|---|---|---|---|---|---|
| A (referencia) | 0.86505 | 0.57674 | 0.17766 | 0.40233 | $522.31 |
| **B: sin el bloque** | **-0.0580** [-0.0623, -0.0538] | **-0.0941** | **+0.0255** | **-0.1413** (-35 %) | **-56.15** (-10.8 %) |
| C: sin derivadas | -0.0001 [-0.0007, +0.0004] | -0.0013 | +0.0002 | -0.0019 | +0.03 |
| D: sin originales | 0.0000 [-0.0004, +0.0004] | +0.0005 | +0.0001 | -0.0010 | +0.48 |

B empeora en las 25 de 25 particiones en todas esas métricas (p < 0.001). C y
D no difieren significativamente de A en ninguna. El error de calibración no
cambia de forma significativa en ningún escenario.

**Confirmación en validación (informativa):**

| Escenario | AUC | KS | Log Loss | Beneficio total | Decisiones distintas a A |
|---|---|---|---|---|---|
| A | 0.86854 | 0.5832 | 0.1764 | $15,777,200 | — |
| B | 0.80933 | 0.4762 | 0.2024 | $14,032,000 (**-$1,745,200**) | 3,899 |
| C | 0.86805 | 0.5820 | 0.1765 | $15,777,200 | 397 |
| D | 0.86924 | 0.5877 | 0.1761 | $15,798,800 | 334 |

**Interpretación.**

1. **Contribución incremental: enorme.** Sin el historial de morosidad, el resto
   de variables no lo compensa: se pierde un 6.7 % de AUC, un 35 % de AP y
   $1.75 millones de beneficio en validación. Es la información más valiosa del
   modelo, coherente con la caída de 0.126 al permutar el bloque (E18).
2. **Redundancia interna: casi total.** Cualquiera de las dos mitades basta: los
   3 conteos sin las derivadas (C) o las 3 derivadas sin los conteos (D).
3. Por eso la importancia individual engañaba. Cada variable del bloque es
   **prescindible por separado** pero **el bloque es imprescindible**. La
   importancia nativa reparte el crédito de forma arbitraria entre sustitutos.

**¿Hay mejora?** No. Ningún escenario mejora a A de forma significativa. El
+0.0007 de AUC de D en validación es ruido: la CV da 0.0000 [-0.0004, +0.0004].

**Decisión.** Sin cambios en `xgboost_v1`. Quitar una mitad del bloque
simplificaría el modelo y concentraría la importancia en menos variables, pero
no mejora ninguna métrica. Por el mismo criterio que E17, un modelo
equivalente no justifica el cambio.

**Lección.** Para variables correlacionadas, la pregunta "¿importa esta
variable?" está mal planteada. Las preguntas correctas son "¿importa este
bloque?" (ablación del bloque) y "¿son sustituibles sus miembros?" (ablación
de cada mitad).

**Reproducibilidad.** `experiments/ablacion_bloque_xgb/ablacion_bloque.py`
(≈3 min con 16 núcleos). Resultados y gráfica en `resultados/`.

---

## Síntesis de lecciones

| # | Lección |
|---|---|
| 1 | Los códigos de sistema invierten la señal. Tres variables tenían valores que parecían extremos y cuyo riesgo real era el **opuesto**. |
| 2 | La ingeniería de variables puede restar. Siete derivadas con lógica sólida no aportaron nada y rompieron la interpretabilidad. |
| 3 | La importancia de variables es del modelo, no de los datos. |
| 4 | Un solo split no permite comparar modelos: ±0.003 de ruido. |
| 5 | El umbral de decisión se deriva de la economía, no se aprende. |
| 6 | Los hiperparámetros del algoritmo importaron mucho menos que los umbrales de limpieza. |
| 7 | Las gráficas del EDA deben usar datos crudos; recortar oculta el hallazgo. |
| 8 | Aplicar una técnica donde no hay nada que medir es relleno. |
| 9 | Una comprobación con exenciones puede ocultar defectos: eximir a los indicadores del control de signos permitió que tres llegaran al modelo con la dirección invertida. |
| 10 | En un scorecard WoE los indicadores de faltante son redundantes por construcción — el bin `FALTANTE` ya los codifica. |
| 11 | Una importancia por permutación ≈ 0 no prueba irrelevancia: la ablación pareada es la prueba directa (E17). |
| 12 | Equivalencia no es superioridad: un criterio de parsimonia debe fijarse antes de ver los resultados, no después (E17). |
| 13 | Con variables redundantes, la importancia se mide por bloques: cada variable de atraso es prescindible por separado, pero quitar el bloque cuesta 0.058 de AUC (E18, E19). |

---

[← Modelado](04_modelado.md) · [Índice](README.md) · [Siguiente: Negocio y regulación →](06_negocio_y_regulacion.md)
