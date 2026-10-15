# Plan de desarrollo — Clasificación de películas (Árbol de decisión + RNA)

## Contexto
Proyecto de la materia IA 2026 (UNS). Hay que clasificar películas en `Mala` / `Buena` / `Muy buena`
con un árbol de decisión y una red neuronal (Pandas, Scikit-learn, Keras). Se evalúa el
preprocesamiento, la experimentación (registrada) y una presentación oral de 10 min.
Entrega: 22/10/2026 23:59 (reentrega 05/11). Hoy es 15/10 → ~7 días.

Decisiones tomadas:
- **Modo guiado**: en cada fase Claude explica los conceptos y las decisiones a tomar, el alumno
  escribe el código que implica decisiones o análisis, y Claude lo revisa. Claude escribe solo el
  boilerplate (encabezados, imports, carga/guardado de datos, `shape`/`head`/`info`).
- **Entorno**: local en el repo (Python 3.12 + venv). El notebook vive en `notebooks/proyecto.ipynb`;
  Claude lo lee directamente para revisarlo. Se commitea al terminar cada fase (historial = evolución
  del trabajo). Colab queda como alternativa para abrirlo/compartirlo (el `.ipynb` es compatible).
- Trabajo hecho por el alumno (o en conjunto con su compañero); no se divide.

## Archivos
- `notebooks/proyecto.ipynb` — notebook único de entrega, con secciones en celdas markdown.
- `docs/experimentos.md` — bitácora de experimentos (fecha, modelo, parámetros, métricas,
  conclusión, próximo paso). Es la materia prima de la parte "evolución de las pruebas" de la oral.
- `output/predicciones_test.csv` — copia del test + columnas `pred_arbol` y `pred_rna`.
- `docs/presentacion.pdf` — diapositivas.

Para cada fase, además del código, el alumno anota en `docs/experimentos.md` **qué decidió y por
qué**, para tenerlo listo en la presentación.

## Fases

### Fase 0 — Setup (día 1)
- Entorno (lo prepara Claude, es mecánico y no es contenido de la materia):
  - `requirements.txt` con `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`,
    `jupyter`, `ipykernel`.
  - `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  - `.gitignore` con `.venv/`, `.ipynb_checkpoints/`, `__pycache__/`.
  - Agregar a `CLAUDE.md` los comandos: activar venv (`source .venv/bin/activate`), abrir el
    notebook (VS Code eligiendo el kernel de `.venv`; alternativa `jupyter notebook notebooks/`), y ejecutar el
    notebook completo sin abrirlo (`jupyter nbconvert --to notebook --execute notebooks/proyecto.ipynb`).
- Editor: **VS Code** con la extensión Jupyter (`ms-toolsai.jupyter`) y el kernel de `.venv`.
- Lo hace el alumno: crear `notebooks/proyecto.ipynb` (VS Code + kernel `.venv`) y cargar los CSV
  con rutas relativas (`../data/movies_train.csv`). Los imports y setups se agregan a medida que
  cada fase los necesita.
- Estructura de secciones: 1 Introducción · 2 Análisis exploratorio · 3 Preprocesamiento ·
  4 Árbol de decisión · 5 RNA · 6 Comparación · 7 Predicciones sobre test · 8 Uso de herramientas de IA.

### Fase 1 — Análisis exploratorio (EDA) (día 1–2)
Conceptos: tipos de variables, valores faltantes, distribución de clases, outliers, correlación.
Tareas: `shape`, `info`, `describe`, `isna().sum()`; distribución de `class` (¿está balanceada?);
histogramas de numéricas (budget/revenue/popularity están muy sesgadas); `0` en budget/revenue
como “desconocido”; cardinalidad de categóricas (`genre` baja; productora, director y actores muy alta);
relación de cada variable con la clase (boxplots por clase).
Salida: lista de problemas detectados → define el preprocesamiento.

### Fase 2 — Preprocesamiento (día 2–3) — *se evalúa directamente*
Conceptos clave: fuga de datos (*data leakage*), split train/validación estratificado,
`Pipeline` y `ColumnTransformer`, codificación one-hot vs ordinal, escalado.
Decisiones a tomar (y justificar en la bitácora):
- Separar `X`/`y`; split estratificado (p. ej. 80/20, `random_state` fijo) **antes** de ajustar
  cualquier transformación.
- Descartar columnas identificadoras (`movieId`, título crudo).
- Ingeniería de atributos: año extraído del título; `budget`/`revenue` = 0 → faltante + indicador
  “dato conocido”; transformación logarítmica; posibles ratios (revenue/budget).
- Faltantes: imputación (mediana para numéricas, “desconocido” para categóricas).
- Alta cardinalidad (nombres de personas/productoras): descartar, quedarse con el top-N y agrupar el
  resto en “Otro”, o usar las variables numéricas de popularidad que ya los resumen.
- `belong_to_collection` → booleano; `genre` → one-hot.
- Etiqueta: mapear clases a 0/1/2 (necesario para Keras).
- El árbol no necesita escalado; la RNA sí (StandardScaler) — explicar por qué en la oral.
- `movies_test_actualizado.csv` trae columna `class`: no usarla para entrenar ni para elegir
  hiperparámetros. Consultar a la cátedra si puede usarse como evaluación final.

### Fase 3 — Árbol de decisión (día 3–4)
Conceptos: entropía/Gini, sobreajuste, profundidad, poda, validación cruzada.
- Línea base: clasificador de clase mayoritaria (`DummyClassifier`) para tener referencia.
- Árbol por defecto → observar sobreajuste (accuracy train vs validación).
- Experimentar en orden y registrar cada paso: `max_depth`, `min_samples_leaf`, `criterion`,
  `class_weight`; luego `GridSearchCV` con CV estratificada.
- Usar `random_state` fijo en el árbol y en la búsqueda de hiperparámetros.
- Métricas: accuracy, F1 macro, matriz de confusión, `classification_report`.
- Interpretación: graficar el árbol (primeros niveles) e importancia de atributos — muy útil para
  las diapositivas.

### Fase 4 — Red neuronal con Keras (día 4–5)
Conceptos: capas densas, ReLU, softmax, función de pérdida `sparse_categorical_crossentropy`,
optimizador Adam, épocas/batch, early stopping, dropout, curvas de aprendizaje.
- Imports de Keras y semilla global (`keras.utils.set_random_seed`) al comienzo de la sección.
- Modelo inicial chico (p. ej. 1 capa oculta) → curvas loss/accuracy train vs validación.
- Experimentar en orden y registrar: neuronas y capas, tasa de aprendizaje, dropout/regularización,
  batch size, `class_weight`; usar `EarlyStopping` con el conjunto de validación.
- Mismas métricas que el árbol para poder compararlos.

### Fase 5 — Comparación y predicción final (día 5–6)
- Tabla comparativa árbol vs RNA (métricas de validación, matrices de confusión, ventajas:
  interpretabilidad vs capacidad).
- Reentrenar los modelos finales con todo el train, aplicar **el mismo pipeline** al test, predecir y
  volver a las etiquetas de texto (`Mala`/`Buena`/`Muy buena`).
- Generar `output/predicciones_test.csv` = test original + `pred_arbol` + `pred_rna`
  (verificar 499 filas, mismo orden).

### Fase 6 — Presentación y entrega (día 6–7)
- Diapositivas (10 min): problema y datos → hallazgos del EDA → preprocesamiento y por qué →
  experimentos del árbol (evolución) → experimentos de la RNA (evolución) → resultados y
  comparación → conclusiones.
- Ensayo con preguntas probables (Claude hace de docente): ¿por qué escalar solo para la RNA?,
  ¿cómo evitaron la fuga de datos?, ¿por qué F1 macro?, ¿qué hace el early stopping?, etc.
- Celda final indicando qué partes se hicieron con herramientas de generación de código/texto
  (requisito del código de honor del enunciado).
- `.zip` con `.ipynb` + `.csv` + `.pdf`; ambos integrantes suben la misma entrega.

## Verificación
- Setup: `.venv/bin/python -c "import pandas, sklearn, tensorflow"` funciona.
- El notebook corre de punta a punta (`jupyter nbconvert --execute`) sin errores y con semillas fijas.
- Ninguna transformación se ajusta sobre datos de validación/test (revisar que los `fit` estén
  solo sobre train).
- El CSV final tiene las 499 filas del test, sus columnas originales y las dos de predicción.
- `docs/experimentos.md` tiene cada experimento con parámetros, resultado y conclusión.
- El alumno puede explicar con sus palabras cada decisión de la bitácora (lo chequeamos en el ensayo).
