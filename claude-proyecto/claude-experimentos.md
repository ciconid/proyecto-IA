# Bitácora de experimentos (versión Claude)

Resultados de `claude-proyecto.ipynb`. Semilla 42; split estratificado 80/20 (10 000 entrenamiento /
2 500 validación). Métrica principal: **F1 macro en validación** (la clase *Muy buena* es solo el 15 %).

## Preprocesamiento

| Decisión | Motivo |
|---|---|
| Año extraído del título | El año no venía como columna. Por sí solo separa poco las clases (medianas 2000 / 2000 / 1997), pero el árbol puede combinarlo con otros atributos |
| `budget`/`revenue`/`runtime` = 0 → faltante, más indicadores `*_conocido` | Son datos desconocidos, no ceros reales (~79 % y ~76 %) |
| `pais_conocido` en vez de `production_company_country` | 90 % de faltantes: no tiene sentido imputar el país |
| Descartar `movieId`, `title`, nombres de productora, intérpretes y director | Identificadores o miles de valores distintos; las columnas numéricas de popularidad ya los resumen |
| `genre` → one-hot con `get_dummies` (faltante = "Desconocido") | 8 valores sin orden natural |
| Mediana para imputar numéricas | Robusta a las colas largas |
| Solo para la RNA: `log(1 + x)` en las numéricas con cola larga y `MinMaxScaler(-1, 1)` | El descenso por gradiente necesita escalas comparables; al árbol no lo afecta |
| Clase como one-hot (`OneHotEncoder`) para la RNA | La salida de la red es un vector de 3 probabilidades |
| Medianas, columnas del one-hot y escalador calculados solo con entrenamiento | Evitar fuga de datos |

## Árbol de decisión

| # | Configuración | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|
| — | Línea base (siempre *Buena*) | 0.462 | 0.462 | 0.211 | 0.211 | Piso de referencia |
| E1 | Por defecto (profundidad 30, 1780 hojas) | 1.000 | 0.662 | 1.000 | 0.624 | Sobreajuste: memoriza el entrenamiento |
| E2 | `max_depth=10` (barrido de 2 a 20) | 0.789 | 0.721 | 0.739 | 0.661 | Limitar la profundidad mejora la generalización |
| E3 | + `min_samples_leaf=50` (barrido 1–100) | 0.730 | 0.718 | 0.678 | 0.666 | Leve mejora, brecha ent–val mucho menor |
| E4 | entropía en vez de Gini | 0.724 | 0.715 | 0.664 | 0.653 | Sin diferencia relevante; se queda Gini |
| E4 | + `class_weight='balanced'` (Gini) | 0.677 | 0.641 | 0.658 | 0.614 | Empeora: recall *Muy buena* 0.40→0.65, pero su precisión 0.65→0.37 y recall *Buena* 0.76→0.52 |
| E5 | Barrido conjunto profundidad × hoja (25 combinaciones) → prof. 10, hoja 50 | 0.730 | 0.718 | 0.678 | 0.666 | **Elegido** (113 hojas). Confirma E3; prof. 10–15 con hoja 50 dan 0.663–0.666 |

## Red neuronal

Todas: capas densas ReLU, salida softmax(3), Adam, `categorical_crossentropy`. Early stopping con
paciencia 15 sobre un 15 % separado del entrenamiento.

| # | Configuración | épocas (mejor) | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|---|
| R1 | [16], 200 épocas fijas | 200 | 0.730 | 0.708 | 0.668 | 0.633 | La pérdida de early stopping toca su mínimo en la época 72 y después sube levemente |
| R2 | [16] + early stopping | 87 (72) | 0.722 | 0.707 | 0.653 | 0.627 | Resultado equivalente con menos de la mitad de las épocas |
| R3 | [64, 32] | 27 (12) | 0.721 | 0.704 | 0.646 | 0.627 | Sobreajusta antes |
| R4 | [64, 32] + dropout 0.3 | 55 (40) | 0.736 | 0.711 | 0.665 | 0.636 | Dropout mejora a R3 |
| R5 | [128, 64, 32] + dropout 0.3 | 38 (23) | 0.734 | 0.708 | 0.671 | 0.643 | **Elegido**, mejora chica |
| R6 | R4 con lr = 0.0001 | 262 (247) | 0.740 | 0.708 | 0.670 | 0.628 | Muy lento, no mejora |
| R6 | R4 con lr = 0.01 | 54 (39) | 0.719 | 0.710 | 0.638 | 0.635 | Similar |
| R7 | R4 con batch 128 | 67 (52) | 0.736 | 0.713 | 0.659 | 0.635 | Similar |
| R8 | R4 + `class_weight` balanced | 29 (14) | 0.665 | 0.647 | 0.638 | 0.616 | Empeora, igual que en el árbol |

Modelo final: R5 reentrenado con todo el train durante 23 épocas (la mejor época de R5). Las diferencias
entre R1 y R7 son chicas (0.627–0.643).

Con `StandardScaler` en lugar de `MinMaxScaler` (probado aparte, no está en el notebook), R4 da 0.645 y
R5 0.643: el escalador del laboratorio no cambia los resultados de forma relevante.

## Comparación (validación)

| Modelo | accuracy | F1 macro | F1 Mala | F1 Buena | F1 Muy buena |
|---|---|---|---|---|---|
| Línea base | 0.462 | 0.211 | 0.000 | 0.632 | 0.000 |
| Árbol | 0.718 | 0.666 | 0.771 | 0.732 | 0.497 |
| Red neuronal | 0.708 | 0.643 | 0.777 | 0.718 | 0.433 |

- *Muy buena*: recall 0.40 en el árbol y 0.33 en la red. De sus 384 casos de validación, 180 (árbol) y
  216 (red) se predicen como *Buena*.
- Atributo más importante del árbol: `popularity` (importancia 0.54, primer corte), seguido de
  `popularity_movies_p1` (0.11).

## Control final con la `class` del test (solo informativo, no usado para decidir)

| Modelo | accuracy | F1 macro | F1 Muy buena |
|---|---|---|---|
| Árbol | 0.682 | 0.638 | 0.500 |
| Red neuronal | 0.686 | 0.589 | 0.312 |

Confirma el orden de validación en F1 macro. La red predice *Muy buena* solo para el 4 % del test
(real: 15 %). Confirmar con la cátedra si corresponde mostrar este resultado.
