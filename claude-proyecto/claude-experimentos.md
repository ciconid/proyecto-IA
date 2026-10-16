# Bitácora de experimentos (versión Claude)

Resultados de `claude-proyecto.ipynb`. Semilla 42; split estratificado 80/20 (10 000 entrenamiento /
2 500 validación). Métrica principal: **F1 macro en validación** (la clase *Muy buena* es solo el 15 %).

## Preprocesamiento

| Decisión | Motivo |
|---|---|
| Año extraído del título | El año no venía como columna; separa algo las clases |
| `budget`/`revenue`/`runtime` = 0 → faltante, más indicadores `*_conocido` | Son datos desconocidos, no ceros reales (~79 % y ~76 %) |
| `pais_conocido` en vez de `production_company_country` | 90 % de faltantes: no tiene sentido imputar el país |
| Descartar `movieId`, `title`, nombres de productora, intérpretes y director | Identificadores o miles de valores distintos; las columnas numéricas de popularidad ya los resumen |
| `genre` → one-hot (faltante = "Desconocido") | 8 valores sin orden natural |
| Mediana para imputar numéricas | Robusta a las colas largas |
| Solo para la RNA: `log(1 + x)` y `StandardScaler` | El descenso por gradiente necesita escalas comparables; al árbol no lo afecta |
| `Pipeline` + `ColumnTransformer`, ajustados solo con entrenamiento | Evitar fuga de datos |

## Árbol de decisión

| # | Configuración | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|
| — | Línea base (clase mayoritaria) | 0.462 | 0.462 | 0.211 | 0.211 | Piso de referencia |
| E1 | Por defecto (profundidad 30, 1780 hojas) | 1.000 | 0.662 | 1.000 | 0.624 | Sobreajuste: memoriza el entrenamiento |
| E2 | `max_depth=10` (barrido de 2 a 20) | 0.789 | 0.721 | 0.739 | 0.661 | Limitar la profundidad mejora la generalización |
| E3 | + `min_samples_leaf=50` (barrido 1–100) | 0.730 | 0.718 | 0.678 | 0.666 | Leve mejora, brecha ent–val mucho menor |
| E4 | entropía en vez de Gini | 0.724 | 0.715 | 0.664 | 0.653 | Sin diferencia relevante |
| E4 | + `class_weight='balanced'` (Gini) | 0.677 | 0.641 | 0.658 | 0.614 | Empeora: recall *Muy buena* 0.40→0.65, pero su precisión 0.65→0.37 y recall *Buena* 0.76→0.52 |
| E5 | `GridSearchCV` 5-fold (100 combinaciones) → entropía, prof. 10, hoja 1 | 0.784 | 0.724 | 0.740 | 0.660 | **Elegido** (F1 en CV: 0.640) |

Se elige E5 porque surge de la validación cruzada (más estable que un solo split). Observación: E3 tiene
una brecha entrenamiento–validación menor.

## Red neuronal

Todas: capas densas ReLU, salida softmax(3), Adam, `sparse_categorical_crossentropy`. Early stopping
con paciencia 15 sobre un 15 % separado del entrenamiento.

| # | Configuración | épocas (mejor) | acc ent | acc val | F1 ent | F1 val | Conclusión |
|---|---|---|---|---|---|---|---|
| R1 | [16], 200 épocas fijas | 200 | 0.739 | 0.703 | 0.687 | 0.643 | La pérdida de validación sube: sobreajuste |
| R2 | [16] + early stopping | 31 (16) | 0.719 | 0.708 | 0.654 | 0.639 | Igual resultado, 6× menos épocas |
| R3 | [64, 32] | 20 (5) | 0.727 | 0.704 | 0.668 | 0.637 | Sobreajusta muy rápido |
| R4 | [64, 32] + dropout 0.3 | 38 (23) | 0.744 | 0.714 | 0.682 | 0.650 | **Elegido** |
| R5 | [128, 64, 32] + dropout 0.3 | 32 (17) | 0.746 | 0.710 | 0.691 | 0.648 | Más capacidad no ayuda |
| R6 | R4 con lr = 0.0001 | 196 (181) | 0.743 | 0.714 | 0.675 | 0.643 | Converge muy lento |
| R6 | R4 con lr = 0.01 | 29 (14) | 0.736 | 0.702 | 0.663 | 0.624 | Inestable |
| R7 | R4 con batch 128 | 57 (42) | 0.744 | 0.714 | 0.672 | 0.639 | Sin mejora |
| R8 | R4 + `class_weight` balanced | 41 (26) | 0.695 | 0.656 | 0.673 | 0.627 | Empeora, igual que en el árbol |

Modelo final: R4 reentrenado con todo el train durante 23 épocas (la mejor época de R4).

## Comparación (validación)

| Modelo | accuracy | F1 macro | F1 Mala | F1 Buena | F1 Muy buena |
|---|---|---|---|---|---|
| Línea base | 0.462 | 0.211 | 0.000 | 0.632 | 0.000 |
| Árbol | 0.724 | 0.660 | 0.775 | 0.742 | 0.463 |
| Red neuronal | 0.714 | 0.650 | 0.783 | 0.723 | 0.443 |

- *Muy buena*: recall 0.34 en ambos modelos; 214 de sus 384 casos de validación se predicen como *Buena*.
- Atributo más importante del árbol: `popularity` (importancia 0.43, primer corte).

## Control final con la `class` del test (solo informativo, no usado para decidir)

| Modelo | accuracy | F1 macro |
|---|---|---|
| Árbol | 0.668 | 0.615 |
| Red neuronal | 0.698 | 0.636 |

En test la red supera al árbol; el árbol elegido era el de mayor brecha entrenamiento–validación.
Confirmar con la cátedra si corresponde mostrar este resultado.
