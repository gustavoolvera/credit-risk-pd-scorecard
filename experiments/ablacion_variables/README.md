# Ablación de variables del modelo H

Experimento aislado (2026-11-01). Documentado en
[docs/05_experimentos.md — E17](../../docs/05_experimentos.md#e17--ablación-de-variables-del-modelo-h).

**Decisión: se mantiene H (10 variables).** Nada de esta carpeta modifica el
pipeline: los scripts solo escriben en `resultados/`.

| Archivo | Contenido |
|---|---|
| `ablacion.py` | Relevancia de las 4 candidatas y ablación A–E en CV 5×5 con WoE por partición (≈80 s) |
| `revision_metodologica.py` | Origen de los cortes, Holm, ingreso y capacidad de pago, no inferioridad con réplica, selección con E como candidato (≈2 min) |
| `resultados/` | CSV, configuración con versiones y gráfica de diferencias pareadas |

Para reproducir, desde la raíz del proyecto:

```
python experiments/ablacion_variables/ablacion.py
python experiments/ablacion_variables/revision_metodologica.py
```

`revision_metodologica.py` lee `resultados/comparaciones_pareadas.csv`, así que
`ablacion.py` debe ejecutarse antes.
