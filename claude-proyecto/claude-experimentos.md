# Bitácora de experimentos (versión Claude)

Resultados de `claude-proyecto.ipynb`. Semilla 42; split estratificado 80/20 (10 000 entrenamiento /
2 500 validación). Métrica principal: **F1 macro en validación** (la clase *Muy buena* es solo el 15 %).

Solo se usan técnicas que aparecen en la teoría o en los laboratorios. La única excepción es el logaritmo
antes de escalar para la red: es una decisión de preprocesamiento que sale del análisis exploratorio (ver
abajo).

## Preprocesamiento

| Decisión | Motivo |
|---|---|
| Año extraído del título | El año no venía como columna. Por sí solo separa poco las clases (promedios 1992 / 1991 / 1990), pero el árbol puede combinarlo con otros atributos |
| `budget`/`revenue`/`runtime` = 0 → faltante, más indicadores `*_conocido` | Son datos desconocidos, no ceros reales (~79 % y ~76 %) |
| `pais_conocido` en vez de `production_company_country` | 90 % de faltantes: no tiene sentido completar el país |
| Descartar `movieId`, `title`, nombres de productora, intérpretes y director | Identificadores o miles de valores distintos; las columnas numéricas de popularidad ya los resumen |
| `genre` → one-hot con `get_dummies` (faltante = "Desconocido") | 8 valores sin orden natural |
| Promedio para completar numéricas | Como `Age` en el laboratorio 1 |
| Solo para la RNA: `log(1 + x)` en las numéricas con cola larga y `MinMaxScaler(-1, 1)` | El descenso por gradiente necesita escalas comparables; sin el logaritmo, unos pocos valores enormes aplastan al resto (la misma red baja de 0.65 a 0.44 de F1 macro). Al árbol no lo afecta |
| Clase como one-hot (`OneHotEncoder`) para la RNA | La salida de la red es un vector de 3 probabilidades |
| Promedios, columnas del one-hot y escalador calculados solo con entrenamiento | Evitar fuga de datos |

## Árbol de decisión

| # | Configuración | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|
| E1 | Por defecto (profundidad 30, 1763 hojas) | 1.000 | 0.667 | 1.000 | 0.624 | Sobreajuste: memoriza el entrenamiento |
| E2 | `max_depth=10` (barrido de 2 a 20) | 0.789 | 0.726 | 0.740 | 0.665 | Limitar la profundidad mejora la generalización |
| E3 | + `min_samples_split=100` (barrido 2–200) | 0.748 | 0.724 | 0.690 | 0.670 | Leve mejora, brecha ent–val mucho menor (0.075 → 0.020). Con 200 subajusta |
| E4 | Barrido conjunto profundidad × mínimo para dividir (25 combinaciones) → prof. 12, mínimo 100 | 0.758 | 0.723 | 0.707 | 0.670 | **Elegido** (162 hojas). Empata con prof. 10; con prof. 6 nada pasa de 0.583 |

`min_samples_split` es uno de los parámetros que nombra el laboratorio 2, que pide usar al menos dos además
de `max_depth`.

## Red neuronal

Todas: capas densas ReLU, salida softmax(3), Adam, `categorical_crossentropy`, batch 32 y tasa 0.001
salvo que se indique. El conjunto de validación se pasa a `fit` para ver las curvas (como en el laboratorio
3); la época con menor pérdida en validación indica cuántas épocas usar.

| # | Configuración | menor pérdida val (época) | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|---|
| R1 | [16], 200 épocas | 200 | 0.725 | 0.714 | 0.661 | 0.644 | **Elegido**. La pérdida de validación se estabiliza desde la época 25; no sobreajusta |
| R2 | [64, 32], 100 épocas | 26 | 0.779 | 0.695 | 0.737 | 0.636 | Sobreajusta desde la época 26 |
| R3 | [64, 32], 26 épocas | 26 | 0.735 | 0.710 | 0.662 | 0.635 | Igual F1, brecha mucho menor |
| R4 | [128, 64, 32], 100 épocas | 4 | 0.831 | 0.662 | 0.800 | 0.594 | Sobreajusta muy rápido |
| R5 | [128, 64, 32], 4 épocas | 4 | 0.711 | 0.710 | 0.619 | 0.619 | No alcanza a aprender |
| R6 | R3 con lr = 0.0001 | 26 | 0.714 | 0.708 | 0.643 | 0.639 | Todavía aprendiendo al final |
| R6 | R3 con lr = 0.01 | 7 | 0.717 | 0.702 | 0.628 | 0.602 | Sobreajusta desde la época 7 |
| R7 | R3 con batch 128 | 26 | 0.726 | 0.706 | 0.653 | 0.628 | Menos actualizaciones, algo peor |

Modelo final: R1 reentrenado con todo el train durante 200 épocas. Las diferencias entre configuraciones
son chicas (0.594–0.644).

## Comparación (validación)

| Modelo | accuracy | F1 macro | F1 Mala | F1 Buena | F1 Muy buena |
|---|---|---|---|---|---|
| Árbol | 0.723 | 0.670 | 0.769 | 0.743 | 0.499 |
| Red neuronal | 0.714 | 0.644 | 0.778 | 0.731 | 0.422 |

- *Muy buena*: recall 0.39 en el árbol y 0.32 en la red.
- Primer corte del árbol: `popularity`.

## Control final con la `class` del test (solo informativo, no usado para decidir)

| Modelo | accuracy | F1 macro | F1 Muy buena |
|---|---|---|---|
| Árbol | 0.704 | 0.669 | 0.553 |
| Red neuronal | 0.696 | 0.622 | 0.400 |

Confirma el orden de validación en F1 macro. La red predice *Muy buena* solo para el 6 % del test
(real: 15 %). Confirmar con la cátedra si corresponde mostrar este resultado.

## Sacado en la simplificación (no aparece en la teoría ni en los laboratorios)

Criterio Gini/entropía, `class_weight`, `min_samples_leaf` (reemplazado por `min_samples_split`), Dropout,
early stopping, línea base, matrices de confusión, importancia de atributos e imputación con la mediana
(reemplazada por el promedio). Los boxplots del análisis exploratorio (2.7) se mantienen.
