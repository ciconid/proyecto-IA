# Guion de la presentación (versión Claude)

10 minutos, ~10 diapositivas. Los gráficos salen de `claude-proyecto.ipynb`.

## Diapositivas

1. **Problema** (30 s): clasificar películas en Mala / Buena / Muy buena; 12 500 películas etiquetadas
   y 500 de test; dos modelos: árbol de decisión y red neuronal.
2. **Los datos** (1 min): tipos de atributos (película, 3 intérpretes, director). Gráfico de
   distribución de clases: *Muy buena* es solo el 15 % → se usa F1 macro.
3. **Problemas encontrados** (1 min): ceros que significan "desconocido" en presupuesto e ingresos
   (~79 % / ~76 %), país faltante en el 90 %, colas largas (histogramas antes/después del logaritmo),
   miles de valores distintos en nombres de personas y productoras.
4. **Preprocesamiento** (1.5 min): tabla de decisiones de `claude-experimentos.md`. Remarcar la
   división estratificada antes de transformar (medianas, one-hot y escalador se calculan solo con
   entrenamiento: fuga de datos) y por qué solo la red usa logaritmo y `MinMaxScaler`.
5. **Árbol: sobreajuste** (1 min): E1 acierta el 100 % del entrenamiento y el 66 % de validación.
   Gráfico F1 vs profundidad.
6. **Árbol: evolución** (1 min): tabla E1→E5; el barrido conjunto de profundidad y mínimo por hoja
   confirma lo que se encontró ajustando un parámetro por vez.
7. **Árbol: interpretación** (45 s): primeros niveles del árbol e importancia de atributos (popularidad).
8. **Red: evolución** (1.5 min): curvas de R1 (sobreajuste sin early stopping) y tabla R1→R8;
   Dropout permite usar redes más grandes; tasa de aprendizaje muy chica = muy lento. Aclarar que las
   diferencias entre configuraciones son chicas.
9. **Comparación** (1 min): tabla de métricas y las dos matrices de confusión; *Muy buena* se confunde
   con *Buena*.
10. **Conclusiones** (45 s): árbol 0.666 y red 0.643 de F1 macro contra 0.21 de la línea base; el árbol
    gana sobre todo en *Muy buena*, y además es interpretable y más simple; más capacidad en la red
    ayudó muy poco → el límite está en los atributos.

## Preguntas probables

- **¿Por qué F1 macro y no solo accuracy?** Con clases desbalanceadas, un modelo que nunca predice
  *Muy buena* puede tener buena accuracy; F1 macro pondera las tres clases por igual.
- **¿Cómo evitaron la fuga de datos?** División antes de cualquier transformación. Las medianas, las
  columnas del one-hot y el escalador se calculan solo con entrenamiento y después se aplican igual a
  validación y test (como el `fit_transform` / `transform` del laboratorio 3).
- **¿Por qué escalar para la red y no para el árbol?** El árbol compara contra umbrales (la escala no
  cambia los cortes); la red aprende por gradiente y las escalas dispares la desestabilizan.
- **¿Por qué descartaron los nombres de actores y directores?** Miles de valores distintos: un one-hot
  crearía miles de columnas casi vacías y el modelo memorizaría nombres. Su popularidad ya está como número.
- **¿Qué hace el early stopping?** Monitorea la pérdida en datos no usados para entrenar y corta cuando
  deja de mejorar, quedándose con los mejores pesos.
- **¿Qué hace el Dropout?** Apaga neuronas al azar durante el entrenamiento; la red no puede depender
  de neuronas puntuales y generaliza mejor.
- **¿Por qué `class_weight` empeoró?** En el árbol, el recall de *Muy buena* sube de 0.40 a 0.65, pero
  el modelo empieza a predecir *Muy buena* de más (precisión 0.65 → 0.37) y el recall de *Buena* cae de
  0.76 a 0.52; el balance neto en F1 macro es negativo.
- **¿Por qué hicieron un barrido conjunto si ya habían ajustado cada parámetro?** Porque la mejor
  profundidad podría cambiar al exigir un mínimo por hoja. El barrido conjunto encontró la misma
  combinación (profundidad 10, hoja 50), lo que da más confianza en la elección.
- **¿Por qué `MinMaxScaler` y no otro escalador?** Es el del laboratorio. Se probó aparte
  `StandardScaler` y los resultados fueron prácticamente iguales (0.643 en ambos para la red elegida).
- **¿Por qué el logaritmo antes de escalar?** Con colas largas, `MinMaxScaler` toma el máximo de unos
  pocos valores enormes y aplasta al resto cerca del -1; el logaritmo comprime esa cola.
