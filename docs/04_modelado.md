# 04 — Modelado: los tres estimadores y sus hiperparámetros

[← Pipeline de datos](03_pipeline_datos.md) · [Índice](README.md) · [Siguiente: Experimentos →](05_experimentos.md)

---

## 4.1 Por qué dos preparaciones de datos incompatibles

**[RECOM]** El error habitual es aplicar un único preprocesamiento a todos los
modelos. Las familias tienen requisitos opuestos:

| Aspecto | Árboles (XGBoost, LightGBM) | Regresión logística |
|---|---|---|
| Outliers | Sin capear | Binning (los absorbe) |
| Faltantes | `NaN` nativo | Bin propio (`FALTANTE`) |
| Escalado | Innecesario | Innecesario (WoE ya normaliza) |
| Codificación | Valores crudos | Weight of Evidence |
| No-linealidad | La aprende sola | Hay que codificarla |
| Multicolinealidad | Indiferente | **Crítica** |
| Variables derivadas | Tolerables | Peligrosas |

### 4.1.1 Fundamento de "no capear para árboles"

**[TEORÍA]** Sea $g$ una transformación estrictamente creciente. Un árbol evalúa
divisiones de la forma $\mathbb{1}\{x \leq c\}$. Como

$$x \leq c \iff g(x) \leq g(c)$$

el conjunto de particiones alcanzables es idéntico. Capear no es una
transformación estrictamente creciente: introduce empates por encima del tope, y
**elimina** la posibilidad de cortar ahí. Es una pérdida estricta de capacidad.

### 4.1.2 Fundamento de "no imputar para árboles"

Ver [§2.6.4](02_fundamentos_matematicos.md#264-manejo-nativo-de-faltantes). El
algoritmo aprende la dirección por defecto que maximiza la ganancia. Con
faltantes informativos (5.57% vs 6.96%), imputar destruye señal.

---

## 4.2 Modelo 1 — Regresión logística sobre WoE

**Script:** `03_train_logistic.py` (~540 líneas, 9 pasos)

### 4.2.1 Rol en el proyecto

Es el **baseline**: el número de referencia contra el que se compara todo lo
demás. Sin baseline, afirmar que un modelo "es bueno" carece de contenido.

Además es el único que produce un **scorecard** (capítulo
[02 §2.11](02_fundamentos_matematicos.md#211-construcción-del-scorecard)).

### 4.2.2 Selección de variables

Se comparan **ocho** subconjuntos, entrenando en 96,000 filas y evaluando en las
24,000 de selección. **La validación no participa.**

**[MEDIDO]**

Con la comprobación **completa** de coherencia —variables WoE contra la
convención de signo **e indicadores contra su dirección univariada medida en los
datos** (ver [H1](07_auditoria.md#h1)):

| Subconjunto | n vars | AUC ajuste | AUC selección | VIF máx | Incoherentes |
|---|---|---|---|---|---|
| A: todas | 24 | 0.86005 | 0.86149 | 80.5 | 6 |
| B: sin `flag_algun`, con `total_atrasos` | 22 | 0.86000 | 0.86126 | 18.1 | 6 |
| C: sin `flag_algun`, con `peor_atraso` | 22 | 0.85950 | 0.86114 | 8.0 | 5 |
| D: sin derivadas de morosidad | 21 | 0.85824 | 0.85908 | 8.0 | 5 |
| E: originales + todos los indicadores | 17 | 0.85874 | 0.86068 | 4.6 | 2 |
| F: sin indicadores de calidad | 18 | 0.85978 | 0.86138 | 58.7 | 4 |
| G: originales + `flag_algun_atraso` | 11 | 0.85877 | 0.86073 | 3.6 | 1 |
| **H: solo las 10 originales** | **10** | 0.85736 | **0.85824** | **1.3** | **0** |

Valores de `logistic_comparacion_subconjuntos.csv` a 2026-11-01 (ver [§4.8.6](#486-scorecard-interpretabilidad-y-negocio-medido)).

**Resultado central:** solo el subconjunto de las 10 variables originales alcanza
coherencia total. Añadir cualquier indicador o variable derivada **desplaza** la
colinealidad en lugar de eliminarla.

**¿Sobra alguna de las 10?** La dirección opuesta (quitar variables de H) se
evaluó con una ablación en CV 5×5 con WoE por partición, réplica confirmatoria y
pruebas de no inferioridad ([05 E17](05_experimentos.md#e17--ablación-de-variables-del-modelo-h)).
Quitar `NumberRealEstateLoansOrLines` empeora el modelo de forma significativa.
Quitar `MonthlyIncome` y `NumberOfOpenCreditLinesAndLoans` da un modelo de 8
variables equivalente, pero no mejor. **Se mantiene H.**

### 4.2.3 Criterio de selección: signos primero, AUC después

```python
inc = signos_incoherentes(m.coef_[0], cols, X_fit, y_fit)
...
validos = comparacion[comparacion["incoherentes"] == 0]
mejor = validos.sort_values(["AUC_seleccion", "VIF_max"],
                            ascending=[False, True]).iloc[0]
```

**Justificación [RECOM]:** un coeficiente que contradice la evidencia univariada
no es un detalle estético. Afirma literalmente "más episodios de mora → menos
riesgo", o le da puntos a favor al grupo con 54.67% de eventos. Ante un
supervisor —o ante el cliente rechazado— es indefendible.

**Coste de la restricción [MEDIDO]:** +0.00386 de AUC de selección (0.45%). El
mejor AUC absoluto lo daba un subconjunto con 8 coeficientes incoherentes.

**Principio subyacente [RECOM]:** en un scorecard WoE la codificación de las
variables originales **ya lo contiene todo** — los faltantes (bin `FALTANTE`),
los extremos (bins de los bordes) y la no-linealidad (el propio binning). Añadir
indicadores de calidad o variables derivadas solo puede introducir redundancia.

### 4.2.4 Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| `penalty` | `l2` (defecto) | Convexa y diferenciable; con $n \gg p$ apenas actúa |
| `C` | `1.0` (defecto) | **Verificado irrelevante**: barrido de 0.001 a 100 mueve el AUC en $6\times10^{-5}$ |
| `solver` | `lbfgs` (defecto) | Cuasi-Newton |
| `tol` | `1e-8` (constante `TOL_LBFGS`) | Con el defecto `1e-4` lbfgs paraba en 15 iteraciones con los β precisos solo a ~1e-2; con `1e-8` converge en 40 y son reproducibles a ~1e-6. Ver [§4.8](#48-cambio-2026-11-01-woe-normalizado-y-tolerancia-del-optimizador) |
| `max_iter` | `2000` | Holgura defensiva; el uso real es de 40 iteraciones |
| `class_weight` | `None` | `'balanced'` distorsiona la PD media de 6.7% a 33% sin ganar AUC |
| `random_state` | `42` | Reproducibilidad |

### 4.2.5 Coeficientes del modelo final [MEDIDO]

Valores de `artifacts/models/logistic_v1.joblib` a 2026-11-01 (WoE normalizado,
`tol=1e-8`, 40 iteraciones).

Intercepto: $\beta_0 = -2.6036$

| Variable | $\beta$ | Signo |
|---|---|---|
| `woe_DebtRatio_limpio` | -0.8574 | ✅ |
| `woe_RevolvingUtilizationOfUnsecuredLines` | -0.6098 | ✅ |
| `woe_NumberRealEstateLoansOrLines` | -0.6056 | ✅ |
| `woe_NumberOfTimes90DaysLate` | -0.5257 | ✅ |
| `woe_NumberOfTime30-59DaysPastDueNotWorse` | -0.4886 | ✅ |
| `woe_age` | -0.3939 | ✅ |
| `woe_NumberOfTime60-89DaysPastDueNotWorse` | -0.3723 | ✅ |
| `woe_NumberOfDependents` | -0.2553 | ✅ |
| `woe_NumberOfOpenCreditLinesAndLoans` | -0.0678 | ✅ |
| `woe_MonthlyIncome` | -0.0214 | ✅ |

**Los diez coeficientes son negativos**, como exige la convención WoE, y ninguno
contradice su dirección univariada.

**[RECOM]** El orden por $|\beta|$ **no** es un orden de importancia: las
columnas WoE comparten unidad (log-odds) pero no dispersión. La comparación
correcta es $\beta_j\,\sigma(\text{WoE}_j)$ y la permutation importance, en
`03b_logistic_interpretabilidad.py` (ver [§4.8.6](#486-scorecard-interpretabilidad-y-negocio-medido)).

### 4.2.6 Diagnósticos [MEDIDO]

```
[1] CONVERGENCIA     40 iteraciones con tol=1e-8 (max_iter=2000)
[2] CALIBRACIÓN      error absoluto medio 0.73 pp
[3] INCERTIDUMBRE    CV 5 folds: 0.85792 ± 0.00359
[4] PARÁMETRO C      de 0.001 a 100 → el AUC no se mueve
```

### 4.2.7 Coste medido de la corrección H1

| Métrica | Antes (18 vars, 3 incoherencias) | Después (10 vars, 0) | Δ |
|---|---|---|---|
| AUC validación | 0.86272 | 0.86059 | -0.00257 |
| AUC (CV) | 0.85948 | 0.85794 | -0.00187 |
| KS | 0.5760 | 0.5696 | -0.0064 |
| Calibración | 0.53 pp | **0.75 pp** | peor |
| VIF máximo | 8.8 | **1.3** | mejor |
| Valor de negocio | $175.33 | $171.95 | -$3.38 |

**Lectura honesta [RECOM]:** la corrección **empeora** las cuatro métricas de
rendimiento. Se acepta porque un scorecard cuya tabla de puntos contradice la
evidencia no sirve para su único propósito —explicar decisiones— por muy alto que
sea su AUC. La contrapartida es un VIF de 1.3: los predictores son
prácticamente ortogonales y los coeficientes, estables.

---

## 4.3 Modelo 2 — XGBoost

**Script:** `04_train_xgboost.py` (442 líneas, 10 pasos)

### 4.3.1 Verificación del formato de entrada [MEDIDO]

```
Dimensiones          : 120,000 × 25
Todas numéricas      : True
Valores infinitos    : 0
Columnas con NaN     : 14 de 25 (203,312 valores)
Nombres problemáticos: ninguno
```

Los `NaN` se conservan **a propósito**: son información.

### 4.3.2 El conjunto de *early stopping*: por qué no se usa validación

```python
X_fit, X_es, y_fit, y_es = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=SEMILLA)
```

**[RECOM]** Si se usara validación para decidir cuándo dejar de añadir árboles,
se estaría *ajustando* a ese conjunto, y el AUC reportado sobre él sería
optimista. Apartar un trozo del propio entrenamiento mantiene validación
intacta.

### 4.3.3 El sobreajuste por defecto [MEDIDO]

```
Con parámetros por defecto:
  AUC ajuste     : 0.92919   ← memorizando
  AUC validación : 0.85592   ← peor que la regresión logística
```

Brecha de **7.3 puntos**. Es el error más común al usar XGBoost: sin control de
complejidad sigue construyendo árboles hasta memorizar el ruido.

Tras el ajuste: `0.87651 / 0.86854` — brecha de 0.8 puntos y +0.013 de AUC.

### 4.3.4 Rejilla de hiperparámetros [MEDIDO]

| `max_depth` | `min_child_weight` | `subsample` | `colsample` | `λ` | Árboles | AUC (early-stop) |
|---|---|---|---|---|---|---|
| 3 | 1 | 1.0 | 1.0 | 1 | 164 | 0.86489 |
| 4 | 1 | 1.0 | 1.0 | 1 | 113 | 0.86464 |
| 5 | 5 | 0.8 | 0.8 | 1 | 102 | 0.86555 |
| **5** | **20** | **0.8** | **0.6** | **1** | **105** | **0.86598** ✅ |
| 7 | 20 | 0.8 | 0.6 | 1 | 44 | 0.86591 |
| 6 | 10 | 0.8 | 0.8 | 5 | 91 | 0.86572 |

**La selección se hace por `auc_es`, no por `auc_val`** — elegir por validación
la convertiría en parte de la decisión.

### 4.3.5 Significado de cada hiperparámetro

| Parámetro | Valor | Significado matemático | Efecto |
|---|---|---|---|
| `max_depth` | 5 | Profundidad máxima; $\leq 2^5 = 32$ hojas | Orden máximo de interacción capturable |
| `min_child_weight` | 20 | $\sum_{k\in I_j} h_k \geq 20$ con $h_k = p_k(1-p_k)$ | Con $p\approx0.07$, ~300 observaciones por hoja |
| `subsample` | 0.8 | Fracción de filas por árbol | Reduce varianza (*bagging* dentro del *boosting*) |
| `colsample_bytree` | 0.6 | Fracción de columnas por árbol | Descorrelaciona árboles |
| `learning_rate` | 0.05 | $\nu$ en $F_M = \sum \nu f_m$ | Bajo = más lento, mejor generalización |
| `n_estimators` | 105 (por *early stopping*) | $M$ | Complejidad total |
| `reg_lambda` | 1 (defecto) | $\lambda$ en $\Omega(f)$ | Encoge los pesos de las hojas |

**Sobre `subsample` y `colsample` [TEORÍA]:** se le oculta información a cada
árbol de forma deliberada. La lógica es la de un comité: 105 evaluadores que
vieron datos parciales distintos producen un consenso más robusto que 105 que
vieron todo y comparten el mismo sesgo.

### 4.3.6 `scale_pos_weight` [MEDIDO]

| Configuración | AUC | PD media predicha |
|---|---|---|
| Sin ajustar | **0.86854** | **6.68%** (real: 6.68%) |
| `scale_pos_weight = 14.0` | 0.86793 | 33.48% |

Descartado: no mejora la discriminación y destruye la calibración.

---

## 4.4 Modelo 3 — LightGBM

**Script:** `05_train_lightgbm.py` (421 líneas, 8 pasos)

### 4.4.1 Diferencia estructural con XGBoost

Ver [§2.6.3](02_fundamentos_matematicos.md#263-lightgbm-crecimiento-por-hojas).
Crecimiento por hojas frente a por niveles; gobierna `num_leaves`, no
`max_depth`.

**Consecuencia práctica [RECOM]:** no se puede copiar la configuración de
XGBoost. `max_depth=5` en XGBoost permite hasta 32 hojas equilibradas; en
LightGBM hay que fijar `num_leaves` explícitamente.

### 4.4.2 Velocidad y calidad de los valores por defecto [MEDIDO]

| | Tiempo | AUC ajuste | AUC validación |
|---|---|---|---|
| LightGBM | **0.4 s** | 0.90697 | **0.86644** |
| XGBoost | 1.7 s | 0.92919 | 0.85592 |

LightGBM es ~4× más rápido y sus valores por defecto están notablemente mejor
calibrados: con `n_estimators=100` sin ajustar ya supera a la regresión
logística, mientras XGBoost por defecto queda por debajo.

### 4.4.3 Rejilla [MEDIDO]

| `num_leaves` | `min_child_samples` | `feature_fraction` | `bagging_fraction` | Árboles | AUC (early-stop) |
|---|---|---|---|---|---|
| **15** | **50** | **0.8** | **0.8** | **90** | **0.86569** ✅ |
| 31 | 20 | 1.0 | 1.0 | 105 | 0.86526 |
| 31 | 100 | 0.6 | 0.8 | 70 | 0.86545 |
| 63 | 20 | 1.0 | 1.0 | 85 | 0.86294 |
| 127 | 20 | 1.0 | 1.0 | 24 | 0.86237 |

**Ganó la configuración más simple probada.** Con crecimiento por hojas la
complejidad se dispara rápido: `num_leaves=127` degrada el AUC en 0.003 y el
*early stopping* corta a 24 árboles porque empieza a sobreajustar de inmediato.

### 4.4.4 Correspondencia de hiperparámetros entre bibliotecas

| LightGBM | XGBoost aproximado | Nota |
|---|---|---|
| `num_leaves` | (sin equivalente directo) | El más importante en LightGBM |
| `min_child_samples` | `min_child_weight` | LightGBM cuenta observaciones; XGBoost suma hessianas |
| `feature_fraction` | `colsample_bytree` | Idéntico |
| `bagging_fraction` + `bagging_freq` | `subsample` | LightGBM requiere `bagging_freq ≥ 1` para activarlo |
| `reg_lambda` | `reg_lambda` | Idéntico |

**[IMPL]** Se fija `bagging_freq=1`; sin él, `bagging_fraction` se ignora
silenciosamente — un error frecuente al migrar configuraciones.

---

## 4.5 El ensemble evaluado y descartado

**[MEDIDO]** Se probó el promedio simple de las tres probabilidades:

| Modelo | AUC val. | KS | Calibración |
|---|---|---|---|
| XGBoost | **0.86854** | **0.58323** | **0.337 pp** |
| LightGBM | 0.86905 | 0.58848 | 0.386 pp |
| Ensemble | 0.86803 | 0.58118 | 0.572 pp |
| LogReg | 0.86059 | 0.57197 | 0.750 pp |

**No ganó.** El promedio ayuda cuando los modelos se equivocan en observaciones
distintas; aquí los tres fallan en las mismas personas (correlación de 0.9928
entre las predicciones de XGBoost y LightGBM). Además empeoró notablemente la
calibración (0.572 pp frente a 0.337 pp del mejor individual), porque promediar
probabilidades de modelos con calibraciones distintas degrada la resultante.

**[RECOM]** Se conserva el código del ensemble porque el intento documentado
vale tanto como el resultado: demuestra que la opción se evaluó, no que se
ignoró.

---

## 4.6 Reentrenamiento final

**[IMPL]** Una vez tomadas **todas** las decisiones (variables, hiperparámetros,
número de árboles), el modelo final se reentrena con las **150,000** filas
etiquetadas.

**Justificación [TEORÍA]:** apartar validación tenía un propósito —permitir
medir sin engañarse— que ya se cumplió. Esos 30,000 registros etiquetados son
información utilizable. Lo que **no** puede hacerse es volver a mirarlos para
decidir algo.

**Detalle técnico [IMPL]:** al reentrenar no hay conjunto de vigilancia, luego
`n_estimators` se fija en el valor que eligió el *early stopping* (105) y se
desactiva `early_stopping_rounds`.

**Comprobación de cordura [MEDIDO]:** el AUC in-sample sobre las filas que antes
eran validación sube a 0.87860. **No es una estimación de rendimiento** —esas
filas ahora forman parte del entrenamiento— sino una verificación de que el
reajuste funcionó.

---

## 4.7 Resumen comparativo

| Criterio | LogReg (WoE) | XGBoost | LightGBM |
|---|---|---|---|
| AUC (CV, 5 folds)¹ | 0.85792 ± 0.00359 | **0.86628 ± 0.00306** | 0.86619 ± 0.00343 |
| Gini | 0.7212 | **0.7374** | 0.7370 |
| KS | 0.5722 | **0.5865** | 0.5832 |
| Average Precision | 0.3883 | **0.4057** | 0.4034 |
| Error de calibración | 0.73 pp | **0.35 pp** | 0.36 pp |
| Interpretabilidad | **Total (scorecard)** | Baja (requiere SHAP) | Baja (requiere SHAP) |
| Tiempo de entrenamiento | Instantáneo | 1.7 s | **0.4 s** |
| Complejidad de despliegue | **Mínima** | Media | Media |
| Riesgo de sobreajuste | **Muy bajo** | Medio | Medio-alto |
| Valor de negocio | $173.21/sol. | $180.37/sol. | **$182.21/sol.** |

La columna LogReg está medida tras el cambio de [§4.8](#48-cambio-2026-11-01-woe-normalizado-y-tolerancia-del-optimizador);
XGBoost y LightGBM no se reentrenaron.

¹ **Media ± desviación estándar de los AUC de los 5 folds, con `np.std()`, es
decir `ddof=0` (divisor n).** No es error estándar ni intervalo de confianza.
Con n = 5, `ddof=0` da una DE un 11 % menor que la muestral (`ddof=1`). Los
valores proceden de los JSON de cada modelo: el de LogReg es la CV propia de
`03` (`tol=1e-8`). La CV comparativa de `05`, con los mismos folds para los
tres modelos, da 0.85794 ± 0.00359 para LogReg. Para comparar modelos, ver
§4.7.1.

### 4.7.1 Incertidumbre de la comparación: diferencias pareadas

**Por qué no basta "media ± DE" [TEORÍA].** Los tres modelos se evalúan en los
**mismos** folds, y la dificultad de cada fold es común a todos: el fold 3 es el
peor para los tres. Barras independientes por modelo ignoran esa covarianza.
Incluso convertidas en IC 95 % (LogReg [0.8530, 0.8629], XGBoost
[0.8620, 0.8705]) se solapan y sugerirían "sin diferencia clara", cuando la
diferencia pareada es nítida. La pregunta "¿es un modelo mejor que otro?" se
responde con las diferencias fold a fold.

**AUC por fold [MEDIDO]** (CV de `05`, reconstruida con el mismo procedimiento;
reproduce exactamente la media y la DE guardadas en `comparacion_modelos_cv.csv`):

| Fold | LogReg (WoE) | XGBoost | LightGBM |
|---|---|---|---|
| 1 | 0.86277 | 0.87095 | 0.87107 |
| 2 | 0.85790 | 0.86495 | 0.86555 |
| 3 | 0.85161 | 0.86179 | 0.86072 |
| 4 | 0.85885 | 0.86577 | 0.86543 |
| 5 | 0.85856 | 0.86796 | 0.86819 |
| Media | 0.85794 | 0.86628 | 0.86619 |
| DE (`ddof=0`, la publicada) | 0.00359 | 0.00306 | 0.00343 |
| DE (`ddof=1`) | 0.00402 | 0.00342 | 0.00383 |

**Procedimiento [IMPL].** t-test con varianza corregida de Nadeau & Bengio
(2003), en la forma de Bouckaert & Frank (2004) para validación cruzada
(`diferencia_pareada_cv` en `src/evaluation.py`):

$$\hat\sigma^2_{\text{corr}} = \left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2, \qquad
\text{IC} = \bar d \pm t_{J-1,\,0.975}\,\hat\sigma_{\text{corr}}$$

**Verificación contra el diseño de `05`:** un solo 5-fold estratificado sobre
150,000 filas (train + validación), así que J = 5 diferencias pareadas. Cada
fold tiene exactamente 120,000 filas de entrenamiento y 30,000 de prueba:
$n_{\text{test}}/n_{\text{train}} = 0.25 = 1/(k-1)$. Varianza corregida
$= (0.20 + 0.25)\,s_d^2$, con $s_d^2$ muestral (`ddof=1`) y t de Student con 4
g.l. El término $n_{\text{test}}/n_{\text{train}}$ corrige que los conjuntos de
entrenamiento de distintos folds comparten el 75 % de sus filas.

**Resultado [MEDIDO]:**

| Comparación | $\bar d$ | IC 95 % corregido | p | IC 95 % sin corregir |
|---|---|---|---|---|
| XGBoost - LogReg | +0.00835 | [+0.00568, +0.01101] | 0.001 | [+0.00657, +0.01012] |
| LightGBM - LogReg | +0.00826 | [+0.00602, +0.01049] | 0.0005 | [+0.00676, +0.00975] |
| XGBoost - LightGBM | +0.00009 | [-0.00111, +0.00129] | 0.84 | [-0.00071, +0.00089] |

**Lectura.**

- Ambos boosters superan a la regresión logística: los IC corregidos excluyen
  el 0 con holgura.
- **XGBoost frente a LightGBM: no se detecta diferencia, y eso no demuestra que
  sean equivalentes.** El IC es compatible con diferencias de hasta ±0.0013 en
  cualquier sentido. Afirmar equivalencia exigiría un margen fijado de antemano
  y una prueba de equivalencia (como en [E17](05_experimentos.md#e17--ablación-de-variables-del-modelo-h)).
- **Limitaciones.** Con 5 folds, $s_d^2$ se estima con 4 grados de libertad y el
  test tiene poca potencia. La corrección de Nadeau–Bengio es una aproximación:
  supone una correlación entre folds de $n_{\text{test}}/(n_{\text{train}}+n_{\text{test}})$.
  Bouckaert & Frank recomiendan 10×10 repeticiones para inferencias más
  estables.

**Figura.** El panel inferior izquierdo de `05_lightgbm_evaluacion.png` mostraba
"media ± DE" por modelo; desde el 2026-11-03 muestra estas diferencias pareadas
con su IC corregido. Solo cambió la gráfica: `comparacion_modelos_cv.csv`, los
JSON y los modelos no cambiaron. Los AUC por fold y la tabla están en
`experiments/ic_comparacion_cv/resultados/`. Desde el 2026-11-03, `05` guarda
además `artifacts/metrics/comparacion_modelos_cv_folds.csv` y `06` calcula las
mismas diferencias a partir de ese archivo.

### 4.7.2 Uso de validación en la elección del modelo

**Regla [IMPL]** (`06_evaluation.py`, paso 4, sin cambios): entre los modelos
cuyo AUC **de validación** queda a menos de 0.002 del mejor, gana el de menor
error de calibración **de validación**.

**[MEDIDO]** Quedan dentro del margen LightGBM (AUC 0.86905, calibración
0.358 pp) y XGBoost (0.86854, 0.352 pp). Gana XGBoost por 0.006 pp de
calibración.

**Consecuencias.**

1. Validación **sí** participa en una decisión: la elección final entre
   modelos. Variables, hiperparámetros y número de árboles se eligen con
   subconjuntos de train.
2. Las cifras de validación del modelo elegido (AUC 0.86854, beneficio
   $15,777,200) llevan un sesgo de selección. Es pequeño, porque se elige entre
   tres modelos con una regla fija.
3. El desempate se decide por una diferencia de calibración que no es
   significativa; la diferencia de AUC entre los boosters tampoco lo es (§4.7.1).
   Elegir LightGBM habría sido igual de defendible.

**Alternativa sin validación [MEDIDO]** (`experiments/seleccion_cv/`). La misma
regla (margen 0.002 de AUC, desempate por menor error de calibración), fijada
antes de ejecutar, pero con métricas de una CV 5×5 dentro de las 120,000 filas
de train (WoE re-aprendido en cada partición para la LogReg):

| Modelo | AUC CV | Calibración CV (pp) | Brier CV | Beneficio CV ($/sol.) |
|---|---|---|---|---|
| XGBoost | 0.86529 | 0.350 | 0.04896 | 522.56 |
| LightGBM | 0.86520 | **0.319** | 0.04897 | 522.93 |
| LogReg (WoE) | 0.85697 | 0.642 | 0.05026 | 513.17 |

Con esta regla gana **LightGBM**. Las diferencias XGBoost - LightGBM, con IC
95 % de Nadeau–Bengio, no son significativas en ninguna métrica: AUC +0.00009
[-0.0006, +0.0008], calibración +0.031 pp [-0.043, +0.106], Brier -0.00001
[-0.00009, +0.00007] y beneficio -0.37 $/sol. [-1.83, +1.08]. Ambos superan a
la LogReg en todas (p < 0.001).

**Lectura.** Que dos versiones de la misma regla elijan modelos distintos
confirma que el desempate entre boosters lo decide el ruido. Los datos no
distinguen a XGBoost de LightGBM. Eso no prueba que sean equivalentes, pero
cualquier elección entre ellos es arbitraria en lo estadístico y despreciable
en lo económico.

**Por qué la regla CV elige LightGBM pese al AUC algo mayor de XGBoost.** En
ambas reglas el AUC solo **filtra**: entran todos los modelos a menos de 0.002
del mejor. XGBoost (0.86529) y LightGBM (0.86520) entran los dos, y desde ahí
el AUC deja de contar; decide la calibración. Lo mismo pasa con la regla
vigente, pero al revés: en validación LightGBM tiene más AUC (0.86905 frente a
0.86854) y pierde por calibración.

**Comparación de las dos reglas [MEDIDO]:**

| | Regla vigente | Regla CV |
|---|---|---|
| Datos | Validación (30,000), una evaluación | CV 5×5 en train (120,000) |
| Filtro | AUC_val ≥ máx - 0.002 | AUC_cv ≥ máx - 0.002 |
| Desempate | menor error de calibración (deciles) | el mismo, promediado en 25 particiones |
| Diferencia de calibración XGB - LGBM | -0.006 pp; bootstrap IC 95 % [-0.148, +0.108] | +0.031 pp; IC 95 % NB [-0.043, +0.106] |
| Estabilidad de la decisión | XGBoost gana en el 50.6 % de 1,000 remuestras | LightGBM mejor calibrado en 19 de 25 particiones (no independientes) |
| Elige | XGBoost | LightGBM |
| Validación tras elegir | contaminada por la selección | conjunto reservado limpio |

Ninguna de las dos decisiones es estadísticamente firme: en validación es,
literalmente, una moneda al aire. AUC, Brier, log loss y beneficio tampoco
separan a los dos modelos en la CV.

**¿Un criterio secundario explícito?** Se evaluaron los candidatos:

| Criterio | XGBoost | LightGBM | ¿Discrimina? |
|---|---|---|---|
| Calibración (puntuación propia: Brier, log loss) | 0.04896 / 0.17756 | 0.04897 / 0.17757 | No |
| Estabilidad entre particiones (DE del AUC) | 0.00402 | 0.00418 | No |
| Complejidad | 97 árboles, 2,450 hojas | 108 árboles, 1,620 hojas | Depende de cómo se mida |
| Variables | 24 | 24 | No |
| Mantenimiento | biblioteca madura, TreeSHAP | ídem | No |

Ningún criterio secundario separa a los modelos de forma inequívoca. La
complejidad favorece a uno u otro según se cuenten árboles u hojas. Escoger
ahora uno de estos criterios, conociendo ya hacia dónde apunta, sería
precisamente diseñar la regla a posteriori.

**Recomendación: adoptar la regla CV tal como está, y con ella LightGBM.** El
argumento no es que LightGBM sea mejor, porque no hay evidencia de ello. Es
que:

1. La regla CV no se diseñó tras ver los resultados: es la regla vigente con un
   único cambio, el origen de las métricas.
2. Ese cambio aplica el principio que el propio proyecto declara (validación no
   participa en decisiones) y que la regla vigente incumplía.
3. Deja validación como conjunto reservado: sus cifras para el modelo elegido
   pasan a ser una estimación sin sesgo de selección.
4. Elegir entre dos modelos indistinguibles es necesariamente arbitrario. Lo
   defendible es que lo arbitrario venga de una regla fijada de antemano, no de
   la preferencia por el modelo ya elegido.

Las cifras de validación de LightGBM (AUC 0.86905, $15,832,400) no pueden
citarse como motivo de la elección: son la evaluación posterior a ella.

**Alternativa también defendible:** mantener XGBoost y la regla vigente,
declarando que la elección se hizo con validación y que el desempate no es
significativo. Su debilidad es que conserva el uso de validación en una
decisión, y que el argumento a su favor ("ya estaba elegido") es en sí un
producto de esa regla.

**Mejora futura (no aplicable ahora):** el error de calibración por deciles es
una métrica ruidosa para desempatar. Para comparaciones futuras convendría
fijar de antemano una puntuación propia (Brier o log loss) como criterio.
Cambiarlo ahora, conociendo los resultados, sería retrospectivo.

**Si se adopta, cambios necesarios:**

- `06` debe calcular la CV de train (hoy en `experiments/seleccion_cv/`) y
  aplicar la regla sobre ella. La CV de `05` no sirve, porque incluye las filas
  de validación.
- Se regeneran `modelo_elegido.json`, el modelo final, la submission, SHAP, la
  sensibilidad de `07`, el modelo por defecto de `predict.py`, y las cifras del
  README y de esta sección.

**Decisión (2026-11-03): se adopta la regla CV.** `06` la aplica con
`src/seleccion.py` y elige LightGBM. La CV del pipeline reproduce partición a
partición la de `experiments/seleccion_cv/` (diferencia 0). XGBoost queda como
alternativa estadísticamente indistinguible; E18, E19, la CV anidada (E11) y
`experiments/shap_xgboost/` son análisis de XGBoost.

**Independencia de validación.** Validación ya no interviene en la elección,
pero sus resultados se examinaron durante el desarrollo: comparaciones entre
modelos, la regla anterior y el análisis de umbrales. No puede considerarse
completamente independiente de todas las decisiones históricas; sus cifras son
la mejor estimación disponible, no una prueba ciega.

### 4.7.3 Modelo de producción, artefactos e inferencia

| Artefacto | Datos de entrenamiento | Quién lo usa |
|---|---|---|
| `artifacts/models/<modelo>_v1.joblib` | train (120,000) | `06` (comparación en validación, SHAP), `07`, `08` |
| `artifacts/models/final/<elegido>_final.joblib` | train + validación (150,000) | submission de `06`, `predict.py` por defecto |
| `preprocessor_v1.joblib` | train | emparejado con los `*_v1` |
| `preprocessor_final.joblib` | train + validación | emparejado con el modelo final |

**Verificación [MEDIDO].** Puntuar `cs-test.csv` con `predict.py` (modelo
final) reproduce la submission de `06` con una diferencia máxima de 5.3e-7 (el
redondeo a 6 decimales de `predict.py`). `07` solo carga los `*_v1`: el modelo
final ya vio las filas de validación.

**Nota sobre la última fila [MEDIDO]:** LightGBM **gana en dinero pese a perder
en AUC**. No es contradictorio: el AUC promedia el rendimiento sobre todos los
umbrales, mientras que el beneficio depende solo de cómo ordena alrededor del
11.76%. LightGBM acierta mejor justo en esa zona.

**[RECOM]** Es otra razón para no elegir modelo únicamente por AUC.

---

## 4.8 Cambio 2026-11-01: WoE normalizado y tolerancia del optimizador

Dos cambios aplicados juntos a la rama de regresión logística. XGBoost y
LightGBM **no** se tocaron.

| Archivo | Cambio |
|---|---|
| `src/woe.py` (`_tabla_woe`) | Suavizado normalizado en los denominadores |
| `03_train_logistic.py` | `tol=TOL_LBFGS = 1e-8` en el modelo final, la CV y el barrido de `C`; `tol_lbfgs` y `n_iter_lbfgs` se guardan en `logistic_v1.json` |
| `tests/test_woe.py` | Cálculo a mano actualizado + test nuevo: las distribuciones suman 1 |

Scripts reejecutados: `02` → `03` → `03b` → `06` → `07` → `08`. `04` y `05` no se
reejecutaron porque su entrada (rama de árboles) es idéntica byte a byte.

### 4.8.1 Fórmula WoE: anterior y nueva

Con $J$ intervalos, $n_{0j}, n_{1j}$ no-eventos y eventos del intervalo $j$, y
$N_0, N_1$ los totales de train ($\alpha = 0.5$):

$$\text{Anterior:}\quad \text{WoE}_j = \ln\frac{(n_{0j}+\alpha)/N_0}{(n_{1j}+\alpha)/N_1}
\qquad\qquad
\text{Nueva:}\quad \text{WoE}_j = \ln\frac{(n_{0j}+\alpha)/(N_0+\alpha J)}{(n_{1j}+\alpha)/(N_1+\alpha J)}$$

**Efecto de la normalización [TEORÍA]:** con la fórmula anterior las
proporciones suavizadas suman $1 + J\alpha/N$ por clase (medido: entre 1.00025
y 1.00062). Con la nueva suman 1 exactamente, así que %eventos y %no-eventos son
distribuciones de probabilidad genuinas y el IV es la divergencia de Jeffreys
entre ellas. Restando las dos fórmulas:

$$\text{WoE}_j^{\text{nueva}} - \text{WoE}_j^{\text{anterior}}
= \ln\frac{N_1+\alpha J}{N_1} - \ln\frac{N_0+\alpha J}{N_0} \equiv \Delta$$

**$\Delta$ no depende de $j$**: es un desplazamiento constante de toda la
columna. En una regresión logística con intercepto (no penalizado en `lbfgs`),
$\sum_k \beta_k (x_k + \Delta_k) + \beta_0 = \sum_k \beta_k x_k + (\beta_0 + \sum_k \beta_k\Delta_k)$,
luego el óptimo tiene **los mismos $\beta$** y el intercepto se desplaza en
$-\sum_k \beta_k \Delta_k$. PD, AUC, AP y KS no cambian.

**[MEDIDO]** Variables del modelo:

| Variable | $J$ | $\Delta$ | Σ proporciones (anterior) | IV anterior | IV nuevo |
|---|---|---|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 8 | 0.000463 | 1.000499 | 1.14634 | 1.14602 |
| age | 10 | 0.000579 | 1.000623 | 0.24365 | 0.24357 |
| NumberOfTime30-59DaysPastDueNotWorse | 5 | 0.000289 | 1.000312 | 0.74456 | 0.74442 |
| DebtRatio_limpio | 6 | 0.000347 | 1.000374 | 0.07281 | 0.07279 |
| MonthlyIncome | 9 | 0.000521 | 1.000561 | 0.07842 | 0.07839 |
| NumberOfOpenCreditLinesAndLoans | 5 | 0.000289 | 1.000312 | 0.06825 | 0.06824 |
| NumberOfTimes90DaysLate | 5 | 0.000289 | 1.000312 | 0.88518 | 0.88499 |
| NumberRealEstateLoansOrLines | 4 | 0.000231 | 1.000249 | 0.05808 | 0.05807 |
| NumberOfTime60-89DaysPastDueNotWorse | 4 | 0.000231 | 1.000249 | 0.60262 | 0.60252 |
| NumberOfDependents | 5 | 0.000289 | 1.000312 | 0.03602 | 0.03601 |

En los parquet WoE la diferencia por columna es constante (desviación ≤ 1e-15)
en train, validación y test. Los cortes de los intervalos no cambiaron.

### 4.8.2 Tolerancia del optimizador

**Problema detectado [MEDIDO]:** con el `tol=1e-4` por defecto, lbfgs se detenía
en **15 iteraciones** antes del óptimo. El valor de la función objetivo
(log-loss · n + ½‖β‖²) era 21972.7881 frente a **21972.6696** del modelo
convergido, y los $\beta$ diferían del óptimo hasta en $9.9\times10^{-3}$.
Consecuencia práctica: cualquier cambio inocuo en los datos, como el
desplazamiento constante de §4.8.1, movía los $\beta$ ~4e-4, por debajo del
cual no eran reproducibles.

**Diagnóstico [MEDIDO]** (entrenando WoE anterior y nuevo con cada tolerancia):

| `tol` | Iteraciones | max \|Δβ\| entre WoE anterior y nuevo | max \|ΔPD\| |
|---|---|---|---|
| 1e-4 | 15 | 3.7e-4 | 2.1e-4 |
| 1e-8 | ~40 | 3.9e-6 | 1.2e-6 |
| 1e-10 | ~42 | 1.7e-6 | 4.7e-7 |

**Decisión:** `TOL_LBFGS = 1e-8` (40 iteraciones, coste despreciable). `06`
hereda la tolerancia al clonar el modelo con `get_params()`. El benchmark de
`src/woe.py` (`benchmark_binning_cv`) es una herramienta de auditoría manual que
no forma parte del pipeline y conserva su configuración.

### 4.8.3 Los cuatro escenarios [MEDIDO]

Validación (30,000 filas). Los escenarios (b) y (c) se entrenaron en memoria
solo para separar los dos efectos.

| Escenario | WoE | `tol` | AUC | AP | KS | Calibración (pp) | PD media |
|---|---|---|---|---|---|---|---|
| (a) respaldo | anterior | 1e-4 | 0.860594 | 0.388208 | 0.571971 | 0.717 | 6.6580 % |
| (b) | nuevo | 1e-4 | 0.860593 | 0.388210 | 0.571971 | 0.716 | 6.6580 % |
| (c) referencia convergida | anterior | 1e-8 | 0.860606 | 0.388321 | 0.572184 | 0.733 | 6.6609 % |
| **(d) actual** | **nuevo** | **1e-8** | **0.860606** | **0.388321** | **0.572184** | **0.733** | **6.6609 %** |

| Efecto | ΔAUC | ΔAP | ΔKS | ΔCalibración |
|---|---|---|---|---|
| Fórmula WoE, (d) - (c) | -3.6e-8 | -1.6e-8 | 0 | 0 |
| Tolerancia, (c) - (a) | +1.3e-5 | +1.1e-4 | +2.1e-4 | +0.016 pp |
| Total, (d) - (a) | +1.3e-5 | +1.1e-4 | +2.1e-4 | +0.016 pp |

**Lectura:** el cambio total se explica por completo por la tolerancia. La
fórmula WoE no tiene efecto medible.

### 4.8.4 Coeficientes e intercepto [MEDIDO]

| Variable | $\beta$ respaldo (a) | $\beta$ referencia (c) | $\beta$ actual (d) | (d) - (c) |
|---|---|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | -0.6120369 | -0.6098067 | -0.6098067 | 0.0e+00 |
| age | -0.3839401 | -0.3938506 | -0.3938520 | -1.4e-06 |
| NumberOfTime30-59DaysPastDueNotWorse | -0.4891036 | -0.4885696 | -0.4885694 | +2.0e-07 |
| DebtRatio_limpio | -0.8652123 | -0.8574126 | -0.8574118 | +8.0e-07 |
| MonthlyIncome | -0.0291809 | -0.0214439 | -0.0214455 | -1.6e-06 |
| NumberOfOpenCreditLinesAndLoans | -0.0728819 | -0.0677905 | -0.0677866 | +3.9e-06 |
| NumberOfTimes90DaysLate | -0.5252682 | -0.5257212 | -0.5257212 | 0.0e+00 |
| NumberRealEstateLoansOrLines | -0.6023471 | -0.6055788 | -0.6055769 | +1.9e-06 |
| NumberOfTime60-89DaysPastDueNotWorse | -0.3726205 | -0.3722929 | -0.3722929 | 0.0e+00 |
| NumberOfDependents | -0.2472181 | -0.2553170 | -0.2553194 | -2.4e-06 |
| **Intercepto** | -2.6060159 | -2.6050798 | **-2.6036477** | |

- **Intercepto:** el valor teórico de (d) es $\beta_0^{(c)} - \sum_k \beta_k^{(c)}\Delta_k = -2.6036476$,
  frente a -2.6036477 medido (diferencia 1e-7).
- **Signos:** los diez siguen siendo negativos. Subconjunto elegido y orden de
  columnas sin cambios (H: solo las 10 originales).
- **Mayor cambio frente al respaldo:** MonthlyIncome (-0.0292 → -0.0214, un 27 %
  relativo). Es la variable con el β más pequeño y peor determinado, y el
  cambio lo produce la tolerancia: (a) → (c) ya da -0.0214.

### 4.8.5 Validación frente a la referencia convergida

Tolerancias fijadas **antes** de ejecutar los scripts:

| Comprobación | Umbral | Frente al respaldo (a) | Frente a la referencia (c) |
|---|---|---|---|
| \|ΔAUC\| | 1e-5 | ❌ 1.3e-5 | ✅ 3.6e-8 |
| \|ΔAP\| | 1e-5 | ❌ 1.1e-4 | ✅ 1.6e-8 |
| max \|Δβ\| | 1e-4 (respaldo), 1e-5 (referencia) | ❌ 9.9e-3 | ✅ 3.9e-6 |
| max \|ΔPD\| individual | 1e-4 (respaldo), 1e-5 (referencia) | ❌ 7.0e-3 en train, 5.4e-3 en val (media 2.4e-4) | ✅ 1.3e-6 (media 6e-8) |
| Error de calibración | ≤ 0.005 pp | ❌ +0.016 pp | ✅ 0 |
| Intercepto vs valor teórico | 1e-5 | — | ✅ 1e-7 |

**Las diferencias frente al respaldo superan los umbrales originales.** Se
aceptaron de forma explícita (decisión del 2026-11-01) porque:

1. Su causa está demostrada: el modelo respaldado no estaba convergido (§4.8.2).
2. Frente a la referencia convergida, que aísla el efecto de la fórmula WoE,
   todas las comprobaciones pasan con 2–3 órdenes de magnitud de margen.
3. En escala estadística son ruido: ΔAUC = 1.3e-5 es el 0.4 % de la desviación
   del AUC entre folds de CV (0.0036).

### 4.8.6 Scorecard, interpretabilidad y negocio [MEDIDO]

**Scorecard (`08`):** la suma de puntos sigue reproduciendo el score del modelo
(diferencia máxima 0.0000000000). Los puntos por atributo se mueven como máximo
0.4. El umbral de aprobación (PD < 800/6,800 = 11.76 %) equivale a 545.26
puntos; se aprueba al 88.2 % (sin cambio).

| Banda | n antes | n después | Default antes | Default después |
|---|---|---|---|---|
| < 500 | 942 | 938 | 54.88 % | 55.01 % |
| 500–550 | 3,357 | 3,354 | 22.07 % | 22.09 % |
| 550–600 | 14,728 | 14,792 | 4.49 % | 4.48 % |
| 600–650 | 10,973 | 10,916 | 0.78 % | 0.78 % |

**Clientes que cambian de banda:** 112 en total (el cambio neto de la banda
550–600 es +64): 75 de 600–650 a 550–600, 18 de 550–600 a 600–650, 11 de
500–550 a 550–600, 4 de 550–600 a 500–550 y 4 de < 500 a 500–550. Su score se
mueve 0.23 puntos de media y 0.52 como máximo.

**¿Alguno cruza el umbral de decisión?** **Ninguno de los 112.** Los cortes de
banda (500/550/600/650) no coinciden con el de aprobación (545.26).

Lo que sí cambia de decisión son **20 clientes que no cambian de banda**, todos
con score entre 545.0 y 545.5 (PD entre 11.67 % y 11.86 %):

| Cambio | Clientes | De ellos, morosos | Efecto en el beneficio |
|---|---|---|---|
| Aprobado → rechazado | 11 | 3 | +3 × $6,000 - 8 × $800 = +$11,600 |
| Rechazado → aprobado | 9 | 0 | +9 × $800 = +$7,200 |
| **Total** | **20** | | **+$18,800** |

Aprobados: 26,459 → 26,457. Los +$18,800 coinciden exactamente con el cambio de
beneficio de LogReg en `comparacion_final.csv` ($15,543,600 → $15,562,400,
+0.12 %). Valor por solicitud: $172.59 → $173.21.

Sensibilidad a otros umbrales: 25 cambios de decisión a PD 10 %, 48 a 5 %, 5 a
20 %. Todos se deben a clientes a una fracción de punto del corte.

**[RECOM]** Que 20 de 30,000 decisiones cambien por una variación numérica de
~2.4e-4 en la PD es inevitable con cualquier umbral duro, y no es una señal de
inestabilidad. El cambio va en la dirección favorable (rechaza a 3 morosos y
aprueba a 9 buenos), pero con n = 20 no es estadísticamente significativo.

**Interpretabilidad (`03b`):**

| Variable | β·σ antes | β·σ después | Rango β·σ | Caída AUC antes | Caída AUC después | Rango AUC |
|---|---|---|---|---|---|---|
| age | -0.2099 | -0.2153 | 5 → **4** | 0.0069 | 0.0071 | 6 → **5** |
| DebtRatio_limpio | -0.2156 | -0.2136 | 4 → **5** | 0.0070 | 0.0070 | 5 → **6** |

El resto de los rangos no cambia. age y DebtRatio están empatados dentro del
ruido de la permutación (σ ≈ 0.001), así que el intercambio no altera la
lectura: RevolvingUtilization y el bloque de morosidad dominan, y DebtRatio pasa
del 1.er lugar por $|\beta|$ bruto al 4.º–5.º al estandarizar.

**Selección de subconjuntos (`03`):** con el optimizador convergido, varios
candidatos tienen **menos** signos incoherentes (A: 8 → 6, B: 8 → 6, C: 7 → 5,
F: 5 → 4). Parte de las "incoherencias" anteriores eran ruido de optimización.
**El elegido no cambia** (H, 0 incoherentes, AUC de selección 0.85824).

**Evaluación de negocio (`06`, `07`):** el modelo elegido por `06` sigue siendo
XGBoost, y las submissions de XGBoost y LightGBM son idénticas a las anteriores. Cambian solo las filas que
dependen de LogReg: LogReg (beneficio +$18,800, KS 0.57197 → 0.57218,
calibración 0.717 → 0.733 pp) y el ensemble que la promedia (beneficio
$15,767,200 → $15,774,000, calibración 0.524 → 0.512 pp).
`business_sensibilidad.csv`, `comparacion_modelos_cv.csv` y
`shap_importancia.csv` son idénticos.

### 4.8.7 Ausencia de fuga de información y tests [MEDIDO]

| Comprobación | Resultado |
|---|---|
| Train y validación sin personas en común | ✅ |
| Mismas columnas y orden en train y validación | ✅ |
| Tablas WoE recalculadas **con train** reproducen los valores guardados | ✅ diferencia 5.0e-13 |
| Tablas WoE recalculadas **con validación** reproducen los valores guardados | ❌ diferencia de hasta 1.99, como debe ser: prueba de que no se ajustaron con validación |
| Todo valor WoE de validación pertenece a las tablas aprendidas en train | ✅ |
| Rama de árboles (`trees/` train, val, test) idéntica | ✅ |
| `xgboost_v1.joblib`, `lightgbm_v1.joblib` sin modificar (md5) | ✅ |
| `pytest tests/` | ✅ 96 passed |

### 4.8.8 Reproducibilidad

- **Respaldo del estado anterior:** `Give Me Some Credit (PD)/backups/2026-11-01_pre_woe_normalizado/`
  (`processed/`, `artifacts/`, `woe.py` original y `MANIFEST_original.md5` con
  36 archivos verificados). Para restaurar: copiar `processed/` a
  `datasets/processed/`, `artifacts/` a `artifacts/` y `woe.py` a `src/woe.py`,
  y comprobar con `md5sum -c MANIFEST_original.md5`.
- **Entorno:** Python 3.13.9, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3,
  matplotlib 3.10.6. Semilla 42 en todo el pipeline.
- **Regeneración:** `python 02_preprocessing.py && python 03_train_logistic.py &&
  python 03b_logistic_interpretabilidad.py && python 06_evaluation.py &&
  python 07_business_evaluation.py && python 08_scorecard.py`. Dos ejecuciones
  consecutivas de `03b` producen archivos idénticos byte a byte.

---

[← Pipeline de datos](03_pipeline_datos.md) · [Índice](README.md) · [Siguiente: Experimentos →](05_experimentos.md)
