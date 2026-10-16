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
Objetivo: tener un entorno reproducible y el notebook listo para trabajar.

- **Entorno** (Claude): `.venv` con `requirements.txt`, `.gitignore`, comandos en `CLAUDE.md`. ✅
  *Por qué:* todos corren las mismas versiones de librerías y el repo no guarda archivos generados.
- **Editor**: VS Code + extensión Jupyter + kernel `.venv`. ✅
  *Por qué:* el notebook vive en el repo y Claude lo puede revisar directo.
- **Estructura del notebook** (Claude): secciones 1–8 como celdas markdown + carga de los CSV. ✅
  *Por qué:* el enunciado pide un notebook bien organizado; las secciones guían el trabajo.
- **Introducción y primera mirada** (alumno): párrafo de introducción y respuestas sobre
  `shape`/`info`.
  *Por qué:* es el primer minuto de la oral y el punto de partida del EDA.
- Los imports y setups se agregan a medida que cada fase los necesita.

### Fase 1 — Análisis exploratorio (EDA) (día 1–2)
Objetivo: conocer los datos antes de modelar. Cada problema que se encuentre acá es una decisión
de preprocesamiento en la Fase 2.

- **Tipos de variables** (`info`): clasificar columnas en numéricas, categóricas y texto/IDs.
  *Por qué:* cada tipo se procesa distinto (escalar, codificar o descartar).
- **Valores faltantes** (`isna().sum()`): cuántos hay y en qué columnas.
  *Por qué:* los modelos no aceptan vacíos; hay que decidir si imputar o descartar.
- **Ceros que significan "desconocido"** en `budget` y `revenue`.
  *Por qué:* son faltantes disfrazados; tratados como 0 real engañan al modelo.
- **Distribución de la clase** (`value_counts` de `class`): ¿hay clases mucho más frecuentes?
  *Por qué:* si está desbalanceada, la accuracy engaña y conviene F1 macro o `class_weight`.
- **Distribución de las numéricas** (`describe`, histogramas): rangos, sesgos y valores extremos.
  *Por qué:* define si hace falta transformación logarítmica y escalado (clave para la RNA).
- **Cardinalidad de las categóricas** (`nunique`): cuántos valores distintos tiene cada una.
  *Por qué:* pocas (ej. `genre`) → one-hot; miles (directores, actores) → no se pueden codificar
  directamente.
- **Relación con la clase** (boxplots o promedios por clase): qué variables cambian según la clase.
  *Por qué:* da intuición de qué atributos van a ser útiles y sirve para las diapositivas.

Salida: celda markdown con la lista de hallazgos, que se anotan también en `docs/experimentos.md`.

### Fase 2 — Preprocesamiento (día 2–3) — *se evalúa directamente*
Objetivo: transformar los datos crudos en una matriz numérica sin faltantes que los modelos
puedan usar, sin filtrar información de validación/test. Cada decisión se justifica en la bitácora.

- **Separar `X`/`y` y hacer split estratificado** (`train_test_split`, 80/20, `stratify`,
  `random_state` fijo) **antes** de cualquier transformación.
  *Por qué:* si imputación o escalado se ajustan con datos de validación, hay fuga de datos y las
  métricas salen infladas. Estratificar mantiene la proporción de clases en ambos conjuntos.
- **Descartar identificadores** (`movieId`, título crudo).
  *Por qué:* son únicos por película; el modelo solo podría memorizarlos, no generalizar.
- **Ingeniería de atributos**: año desde el título; `budget`/`revenue` = 0 → faltante + indicador
  "dato conocido"; logaritmo en montos; posibles ratios (revenue/budget).
  *Por qué:* extrae información útil escondida en el texto y corrige los faltantes disfrazados y
  los sesgos detectados en el EDA.
- **Imputar faltantes** (`SimpleImputer`: mediana en numéricas, "desconocido" en categóricas).
  *Por qué:* los modelos no aceptan vacíos; la mediana es robusta a valores extremos.
- **Alta cardinalidad** (productoras, directores, actores): descartar, top-N + "Otro", o apoyarse en
  las columnas numéricas de popularidad.
  *Por qué:* un one-hot con miles de valores genera miles de columnas casi vacías.
- **Codificar categóricas**: `belong_to_collection` → booleano; `genre` → one-hot.
  *Por qué:* los modelos operan con números; one-hot evita inventar un orden entre géneros.
- **Codificar la etiqueta** a 0/1/2.
  *Por qué:* Keras necesita la clase como entero para `sparse_categorical_crossentropy`.
- **Escalar solo para la RNA** (`StandardScaler`).
  *Por qué:* el árbol compara umbrales y la escala no le afecta; la RNA usa descenso por gradiente
  y escalas muy distintas dificultan el entrenamiento. Pregunta probable en la oral.
- **Armarlo con `Pipeline` + `ColumnTransformer`**.
  *Por qué:* garantiza que train, validación y test pasen por exactamente las mismas transformaciones.
- **No usar la `class` del test** para entrenar ni elegir hiperparámetros. Consultar a la cátedra
  si puede usarse como evaluación final.
  *Por qué:* el test simula datos nunca vistos; usarlo para decidir invalida la evaluación.

### Fase 3 — Árbol de decisión (día 3–4)
Objetivo: entrenar y ajustar un árbol, registrando la evolución de los experimentos.

- **Línea base** (`DummyClassifier`, clase mayoritaria).
  *Por qué:* da el piso mínimo; un modelo que no lo supera no aprendió nada.
- **Árbol por defecto** y comparar accuracy en train vs validación.
  *Por qué:* un árbol sin límites memoriza el train; ver el sobreajuste justifica ajustar parámetros.
- **Experimentar de a un hiperparámetro**: `max_depth`, `min_samples_leaf`, `criterion`,
  `class_weight`, anotando cada prueba en la bitácora.
  *Por qué:* el enunciado pide mostrar con qué se empezó, qué se varió y en qué orden.
- **Búsqueda sistemática** (`GridSearchCV` con CV estratificada).
  *Por qué:* la validación cruzada da una estimación más estable que un solo split.
- **`random_state` fijo** en el árbol y en la búsqueda.
  *Por qué:* resultados reproducibles y comparables entre experimentos.
- **Métricas**: accuracy, F1 macro, matriz de confusión, `classification_report`.
  *Por qué:* la matriz muestra qué clases se confunden; F1 macro no se deja engañar por el desbalance.
- **Interpretación**: graficar los primeros niveles del árbol e importancia de atributos.
  *Por qué:* es la gran ventaja del árbol y material muy visual para las diapositivas.

### Fase 4 — Red neuronal con Keras (día 4–5)
Objetivo: entrenar y ajustar una RNA, registrando la evolución de los experimentos.

- **Imports de Keras y semilla global** (`keras.utils.set_random_seed`) al inicio de la sección.
  *Por qué:* los pesos iniciales son aleatorios; sin semilla cada corrida da resultados distintos.
- **Modelo inicial chico** (1 capa oculta densa + ReLU, salida softmax de 3 neuronas, Adam,
  `sparse_categorical_crossentropy`).
  *Por qué:* empezar simple da una referencia; softmax produce una probabilidad por clase.
- **Curvas de aprendizaje** (loss y accuracy, train vs validación, por época).
  *Por qué:* muestran si el modelo aprende, si sobreajusta (validación empeora) o si le falta capacidad.
- **Experimentar de a un cambio**: neuronas y capas, tasa de aprendizaje, dropout/regularización,
  batch size, `class_weight`, anotando cada prueba.
  *Por qué:* igual que con el árbol, la evolución de las pruebas es parte de la evaluación.
- **`EarlyStopping`** sobre la pérdida de validación.
  *Por qué:* corta el entrenamiento cuando deja de mejorar y evita sobreajustar sin adivinar las épocas.
- **Mismas métricas que el árbol**.
  *Por qué:* permite compararlos de forma justa en la Fase 5.

### Fase 5 — Comparación y predicción final (día 5–6)
Objetivo: elegir las versiones finales y generar el CSV de entrega.

- **Tabla comparativa árbol vs RNA** (métricas de validación, matrices de confusión).
  *Por qué:* resume los resultados para la oral; incluir ventajas cualitativas (interpretabilidad
  vs capacidad).
- **Reentrenar los modelos finales con todo el train**.
  *Por qué:* elegidos los hiperparámetros, más datos dan un modelo un poco mejor.
- **Aplicar el mismo pipeline al test y predecir** (sin volver a ajustarlo).
  *Por qué:* el test debe transformarse con lo aprendido del train, igual que validación.
- **Volver a las etiquetas de texto** (`Mala`/`Buena`/`Muy buena`).
  *Por qué:* el CSV de entrega tiene que ser legible y coincidir con las clases originales.
- **Generar `output/predicciones_test.csv`** = test original + `pred_arbol` + `pred_rna`
  (verificar 500 filas, mismo orden).
  *Por qué:* es un entregable obligatorio con ese formato exacto.

### Fase 6 — Presentación y entrega (día 6–7)
Objetivo: comunicar el trabajo en 10 minutos y entregar el zip completo.

- **Diapositivas**: problema y datos → hallazgos del EDA → preprocesamiento y por qué →
  experimentos del árbol → experimentos de la RNA → resultados y comparación → conclusiones.
  *Por qué:* las diapositivas son el único documento que ve la comisión para evaluar la resolución.
- **Ensayo con preguntas** (Claude hace de docente): ¿por qué escalar solo para la RNA?, ¿cómo
  evitaron la fuga de datos?, ¿por qué F1 macro?, ¿qué hace el early stopping?
  *Por qué:* hay 5 minutos de preguntas y hay que defender cada decisión con palabras propias.
- **Celda final sobre uso de herramientas de IA**.
  *Por qué:* lo exige el código de honor del enunciado.
- **Zip con `.ipynb` + `.csv` + `.pdf`**; ambos integrantes suben la misma entrega.
  *Por qué:* si las entregas difieren, la cátedra corrige solo una.

## Verificación
- Setup: `.venv/bin/python -c "import pandas, sklearn, tensorflow"` funciona.
- El notebook corre de punta a punta (`jupyter nbconvert --execute`) sin errores y con semillas fijas.
- Ninguna transformación se ajusta sobre datos de validación/test (revisar que los `fit` estén
  solo sobre train).
- El CSV final tiene las 500 filas del test, sus columnas originales y las dos de predicción.
- `docs/experimentos.md` tiene cada experimento con parámetros, resultado y conclusión.
- El alumno puede explicar con sus palabras cada decisión de la bitácora (lo chequeamos en el ensayo).
