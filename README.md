# Clasificación de películas: árbol de decisión y red neuronal

Proyecto de Aprendizaje Automático de Inteligencia Artificial 2026 (DCIC, UNS). Clasifica películas
en **Mala**, **Buena** o **Muy buena** con un árbol de decisión (Scikit-learn) y una red neuronal
(Keras). El enunciado está en `docs/`.

## Estructura

| Ruta | Contenido |
|---|---|
| `data/` | `movies_train.csv` (datos etiquetados) y `movies_test_actualizado.csv` (test) |
| `notebooks/proyecto.ipynb` | Notebook del proyecto |
| `claude-proyecto/` | Versión de referencia generada por Claude: notebook, bitácora de experimentos, predicciones y guion de la presentación |
| `docs/` | Enunciado del proyecto |
| `PLAN.md` | Plan de desarrollo por fases |
| `requirements.txt` | Librerías de Python necesarias |

## Requisitos

- **Python 3.12** (es la versión con la que se probó). Verificalo con `python3 --version`.
- ~2 GB libres en disco (TensorFlow es pesado).
- **VS Code** con la extensión **Jupyter** de Microsoft (`ms-toolsai.jupyter`), o bien Jupyter en el
  navegador (ver más abajo).

No hace falta GPU: el dataset es chico y todo corre en CPU en pocos minutos.

## Instalación

1. **Clonar el repo y entrar a la carpeta:**

   ```bash
   git clone <url-del-repo>
   cd proyecto-IA
   ```

2. **Crear el entorno virtual** (una carpeta `.venv/` con un Python propio del proyecto, para no mezclar
   librerías con las del sistema; está en `.gitignore`, así que cada uno lo crea en su máquina):

   ```bash
   python3 -m venv .venv
   ```

   En Ubuntu/Debian, si da error, instalá antes el paquete: `sudo apt install python3-venv`.

3. **Activarlo e instalar las librerías:**

   ```bash
   source .venv/bin/activate        # Linux / macOS
   .venv\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

   Al activarlo, el prompt de la terminal muestra `(.venv)` adelante. Hay que activarlo en cada
   terminal nueva en la que se vaya a usar `python` o `jupyter`.

4. **Verificar la instalación:**

   ```bash
   python -c "import pandas, sklearn, tensorflow; print('ok')"
   ```

## Abrir los notebooks

### Opción A: VS Code (recomendada)

1. Abrir la carpeta del repo: `File → Open Folder…`.
2. Abrir el notebook (por ejemplo `notebooks/proyecto.ipynb`).
3. **Elegir el kernel a mano** (VS Code no lo elige solo): arriba a la derecha, **Select Kernel →
   Python Environments → `.venv`**. Si no aparece, **Select Another Kernel → Python Environments →
   Enter interpreter path…** y escribir la ruta a `.venv/bin/python` (en Windows,
   `.venv\Scripts\python.exe`).
4. Si VS Code ofrece instalar `ipykernel`, aceptar.
5. Ejecutar todo con **Run All**.

El kernel elegido queda guardado por notebook: hay que repetir el paso 3 la primera vez que se abre
cada uno (`notebooks/proyecto.ipynb` y `claude-proyecto/claude-proyecto.ipynb`).

Si una celda da `ModuleNotFoundError: No module named 'pandas'`, el kernel elegido no es el de `.venv`.

### Opción B: Jupyter en el navegador

Con el entorno activado:

```bash
jupyter lab
```

Se abre una pestaña del navegador; navegar hasta el notebook. Usa automáticamente el Python del
entorno activado. La terminal tiene que quedar abierta mientras se trabaja (`Ctrl+C` para cerrar).

### Opción C: Google Colab

Subir el `.ipynb` a Colab y los dos CSV de `data/` desde el panel de archivos. Hay que cambiar las
rutas de `pd.read_csv('../data/...')` por `pd.read_csv('movies_train.csv')` (o donde se hayan subido).

## Ejecutar un notebook completo sin abrirlo

Con el entorno activado, desde la raíz del repo:

```bash
jupyter nbconvert --to notebook --execute --inplace claude-proyecto/claude-proyecto.ipynb
```

Ejecuta todas las celdas y guarda los resultados en el mismo archivo. El notebook de
`claude-proyecto/` tarda unos 3 minutos (entrena varias redes) y regenera
`claude-proyecto/claude-predicciones_test.csv`. Los resultados son reproducibles: usa semillas fijas.

Los notebooks leen los datos con rutas relativas (`../data/`), así que tienen que ejecutarse desde su
propia carpeta. VS Code, `jupyter lab` y `nbconvert` lo hacen solos.
