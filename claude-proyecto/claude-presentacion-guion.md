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
   división estratificada antes de transformar (promedios, one-hot y escalador se calculan solo con
   entrenamiento: fuga de datos) y por qué solo la red usa logaritmo y `MinMaxScaler`.
5. **Árbol: sobreajuste** (1 min): E1 acierta el 100 % del entrenamiento y el 67 % de validación.
   Gráfico F1 vs profundidad.
6. **Árbol: evolución** (1 min): tabla E1→E4; el barrido conjunto de profundidad y mínimo para dividir
   confirma lo que se encontró ajustando un parámetro por vez.
7. **Árbol: interpretación** (45 s): primeros niveles del árbol (el primer corte es la popularidad).
8. **Red: evolución** (1.5 min): curvas de R2 y R4 (sobreajuste: la pérdida de validación sube) y tabla
   R1→R7; las redes grandes sobreajustan en pocas épocas y la red chica es la mejor. Aclarar que las
   diferencias entre configuraciones son chicas.
9. **Comparación** (1 min): tabla de métricas; *Muy buena* es la clase difícil para los dos.
10. **Conclusiones** (45 s): árbol 0.670 y red 0.644 de F1 macro; el árbol gana sobre todo en
    *Muy buena*, y además es interpretable y más simple; más capacidad en la red no ayudó → el límite
    está en los atributos.

## Preguntas probables

- **¿Por qué F1 macro y no solo accuracy?** Con clases desbalanceadas, un modelo que nunca predice
  *Muy buena* puede tener buena accuracy; F1 macro pondera las tres clases por igual (es la fila
  `macro avg` de `classification_report`).
- **¿Cómo evitaron la fuga de datos?** División antes de cualquier transformación. Los promedios, las
  columnas del one-hot y el escalador se calculan solo con entrenamiento y después se aplican igual a
  validación y test (como el `fit_transform` / `transform` del laboratorio 3).
- **¿Por qué escalar para la red y no para el árbol?** El árbol compara contra umbrales (la escala no
  cambia los cortes); la red aprende por gradiente y las escalas dispares la desestabilizan.
- **¿Por qué el logaritmo antes de escalar?** Con colas largas, `MinMaxScaler` toma el máximo de unos
  pocos valores enormes y aplasta al resto cerca del -1; el logaritmo comprime esa cola. Sin él, la misma
  red baja de 0.65 a 0.44 de F1 macro.
- **¿Por qué descartaron los nombres de actores y directores?** Miles de valores distintos: un one-hot
  crearía miles de columnas casi vacías y el modelo memorizaría nombres. Su popularidad ya está como número.
- **¿Cómo eligieron las épocas?** Mirando la curva de pérdida en validación: cuando empieza a subir
  mientras la de entrenamiento sigue bajando, la red está sobreajustando. Se usa la época con menor
  pérdida en validación.
- **¿Qué hace `min_samples_split`?** Un nodo solo se divide si tiene al menos esa cantidad de ejemplos;
  evita reglas basadas en muy pocos casos.
- **¿Por qué hicieron un barrido conjunto si ya habían ajustado cada parámetro?** Porque la mejor
  profundidad podría cambiar al exigir un mínimo para dividir. El barrido conjunto encontró el mismo
  mínimo (100), con profundidad 10 y 12 empatadas, lo que da más confianza en la elección.
