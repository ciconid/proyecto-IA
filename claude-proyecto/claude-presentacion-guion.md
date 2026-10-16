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
   división estratificada antes de transformar (fuga de datos) y por qué solo la red se escala.
5. **Árbol: sobreajuste** (1 min): E1 acierta el 100 % del entrenamiento y el 66 % de validación.
   Gráfico F1 vs profundidad.
6. **Árbol: evolución** (1 min): tabla E1→E5; búsqueda en grilla con validación cruzada.
7. **Árbol: interpretación** (45 s): primeros niveles del árbol e importancia de atributos (popularidad).
8. **Red: evolución** (1.5 min): curvas de R1 (sobreajuste sin early stopping) y tabla R1→R8;
   Dropout como la mejora que funcionó; tasa de aprendizaje muy chica / muy grande.
9. **Comparación** (1 min): tabla de métricas y las dos matrices de confusión; *Muy buena* se confunde
   con *Buena*.
10. **Conclusiones** (45 s): ambos ~0.65 de F1 macro contra 0.21 de la línea base; árbol interpretable
    y más simple; más capacidad en la red no ayudó → el límite está en los atributos.

## Preguntas probables

- **¿Por qué F1 macro y no solo accuracy?** Con clases desbalanceadas, un modelo que nunca predice
  *Muy buena* puede tener buena accuracy; F1 macro pondera las tres clases por igual.
- **¿Cómo evitaron la fuga de datos?** División antes de cualquier transformación y todo dentro de un
  `Pipeline` ajustado solo con entrenamiento; el test solo se transforma.
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
- **¿Por qué eligieron el árbol de la grilla si E3 era igual o mejor en validación?** La validación
  cruzada promedia 5 particiones y es más confiable que un único split; la diferencia (0.006) está
  dentro de la variación esperable.
