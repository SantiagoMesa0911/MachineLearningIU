# S30 - EA2 Unidad 1 - Modelos de Predicción y Clasificación

Proyecto que obtiene un dataset de Hugging Face, lo preprocesa, entrena una **regresión lineal**
y un **modelo de clasificación** sobre el mismo dataset, y genera automáticamente un resumen
`.txt` de ambos. Todo el flujo se ejecuta en **GitHub Actions** con cada push.

```
Hugging Face → DataFrame → preprocesamiento
                         → regresión lineal + evaluación
                         → clasificación + evaluación
                         → resumen automático .txt
```

## 👥 Integrantes

- Santiago Mesa Parra
- Tatiana Orozco Alzate

## 📊 Dataset

| Campo | Valor |
|---|---|
| Identificador | `manuelgscom/dale-bash-test-v2` |
| Enlace | https://huggingface.co/datasets/manuelgscom/dale-bash-test-v2 |
| Split / configuración | `test` (único split publicado; configuración `default`) |
| Contenido | 471 peticiones en lenguaje natural (español 257 / inglés 214) a un asistente de terminal Linux |
| Columnas | `id`, `lang`, `category`, `context`, `request`, `expected_type`, `reference_commands`, `notes` |

Columnas relevantes:
- `request`: texto de la petición.
- `expected_type`: si el asistente debe responder con un comando (`command`), rechazar la petición (`refuse`) o pedir aclaración (`unsure`).
- `context`: diccionario con distribución, gestor de paquetes y herramientas instaladas.
- `reference_commands`: los comandos bash correctos.

Se descarga con `datasets.load_dataset()` desde el código (`src/data_loader.py`). No hay
archivos manuales.

- **Clasificación:** objetivo `expected_type`, que es categórico y viene así en el dataset.
- **Regresión:** el dataset no trae ninguna variable numérica continua. Se **construye**
  `command_length`, la longitud en caracteres del comando de referencia, como medida de la
  complejidad de la petición. Solo se usa como objetivo; `reference_commands` nunca entra como predictor.
- **Fuga de información detectada:** `category` equivale a la clase (`fuera_de_ambito` siempre es
  `refuse` y `ambigua` siempre es `unsure`), así que se excluye del clasificador.

## 🗂️ Estructura

```text
S30_Unidad1_EA2_Modelos_de_Prediccion_y_Clasificacion/
├── main.py                          # Punto de entrada del flujo completo (lo usa GitHub Actions)
├── src/
│   ├── config.py                    # Dataset, semilla, test_size, variables de cada modelo
│   ├── data_loader.py               # HFDatasetLoader: Hugging Face -> pandas.DataFrame
│   ├── preprocessing.py             # DataPreprocessor (limpieza y enriquecimiento) y split_data
│   ├── regression_model.py          # LinearRegressionModel (Pipeline + métricas + coeficientes)
│   ├── classification_model.py      # TextLogisticClassifier (TF-IDF + regresión logística)
│   ├── interpretation.py            # Interpretaciones en texto a partir de los resultados
│   ├── pipeline.py                  # run_experiment(): orquesta todo el flujo
│   └── report_writer.py             # ModelSummaryWriter: genera el resumen .txt
├── notebooks/
│   └── modelos_prediccion_clasificacion.ipynb   # Evidencias y explicaciones (ejecutado)
├── output/
│   └── resumen_modelos.txt          # Copia del resumen generado
├── requirements.txt
└── .gitignore

.github/workflows/s30_modelos_ml.yml # (en la raíz del repositorio) workflow de GitHub Actions
```

## 🧹 Preprocesamiento

Las operaciones que **no aprenden de los datos** se aplican al DataFrame completo
(`DataPreprocessor`):

| Operación | Problema que resuelve | Efecto |
|---|---|---|
| Expandir `context` | Es un diccionario anidado | Columnas `os`, `distro`, `pm`, `tools` |
| Eliminar `id`, `os`, `notes` | `id` es un identificador, `os` es constante (`linux`) y `notes` anticipa la respuesta | Columnas sin ruido ni fuga |
| Nulos y duplicados | Una petición repetida en train y test inflaría las métricas | 0 nulos, 0 duplicados |
| Corregir tipos | `lang`, `category`, `distro`, `pm` y `expected_type` eran texto | Tipo `category` |
| Enriquecer | El texto y las listas no son numéricos | `request_length`, `n_paths` (rutas citadas), `n_tools`, `command_length` |
| Eliminar listas resumidas | `tools` y `reference_commands` no son utilizables | 471 filas × 10 columnas |

Las operaciones que **sí aprenden de los datos** van dentro de un `Pipeline` de scikit-learn,
de modo que se ajustan **solo con entrenamiento** y luego se aplican a prueba:

- `StandardScaler` en las variables numéricas, para ambos modelos.
- `OneHotEncoder(drop="first")` en `lang` y `category`, solo en la regresión.
- `TfidfVectorizer` sobre `request`, solo en la clasificación. El vocabulario se aprende con train.

La división es 70 % entrenamiento y 30 % prueba con `random_state=42`. La regresión estratifica
por `category` y la clasificación por `expected_type`.
El notebook demuestra que las medias del escalador son las de `X_train` y que las palabras que
solo aparecen en prueba se ignoran.

## 📈 Modelo 1: Regresión lineal

- **Pregunta:** ¿se puede estimar qué tan largo (complejo) será el comando que resuelve una
  petición a partir de rasgos de la propia petición?
- **Datos:** las 339 peticiones `command` (237 para entrenamiento y 102 para prueba).
- **Objetivo:** `command_length`, en caracteres.
- **Predictoras:** `request_length`, `n_paths`, `n_tools`, `lang` y `category`.
- **Algoritmo:** `LinearRegression`, con todos sus parámetros predeterminados.
- **Resultados:** R² = 0.24 y MAE = 14.8 caracteres, frente a 17.2 de un modelo base que predice
  siempre la media. Además se calcula el R² con validación cruzada de 5 pliegues.
- **Interpretación:**
  - Cada ruta mencionada añade unos 8 caracteres al comando.
  - Cada carácter de la petición añade unos 0.26.
  - La categoría `paquetes` produce comandos unos 12 caracteres más cortos.
  - El R² es modesto porque la longitud depende de la sintaxis concreta de cada comando.

## 🎯 Modelo 2: Clasificación

- **Qué se clasifica:** `expected_type`, con las clases `command` (72 %), `refuse` (21 %) y `unsure` (7 %).
- **Predictoras:** el texto `request` (TF-IDF de palabras y bigramas), más `request_length` y `n_paths`.
  Se excluyen `category`, `reference_commands` y `command_length` por fuga de información.
- **Algoritmo:** TF-IDF más `LogisticRegression`. Es lineal, rápido, adecuado para textos cortos
  con pocos datos, e interpretable, porque cada término tiene un peso por clase.
- **Hiperparámetros:**
  - `class_weight="balanced"`: las clases están desbalanceadas; sin balanceo, el F1 macro cae de
    0.70 a 0.42 y el modelo nunca predice `unsure`.
  - `max_iter=1000` y `random_state=42`.
  - El resto son predeterminados: `C=1.0`, `solver="lbfgs"`, L2.
  - TF-IDF: `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
- **Métricas:** matriz de confusión, recall y F1 por clase y F1 macro. Se prioriza el F1 macro
  porque la exactitud premia a la clase mayoritaria. También hay validación cruzada estratificada.
- **Resultados:**
  - F1 macro = 0.70 y exactitud = 0.82.
  - `command` se reconoce mejor (F1 = 0.90).
  - `unsure` es la clase más difícil (F1 = 0.44, con solo 9 casos en prueba).

Los resultados numéricos y su interpretación están en el notebook y en `output/resumen_modelos.txt`.

## 💻 Ejecución local

Desde esta carpeta:

```bash
python -m venv .venv
```

Activar el entorno (Windows: `.venv\Scripts\Activate.ps1`; Linux/macOS: `source .venv/bin/activate`):

```bash
pip install -r requirements.txt
```

Flujo completo (genera `output/resumen_modelos.txt`):

```bash
python main.py
```

Notebook de evidencias:

```bash
jupyter notebook notebooks/modelos_prediccion_clasificacion.ipynb
```

Se requiere Internet para descargar el dataset la primera vez.

## ⚙️ Ejecución automática (GitHub Actions)

El workflow `.github/workflows/s30_modelos_ml.yml` se activa con cada **push** a `main` que
modifique esta carpeta o el propio workflow. También se puede lanzar manualmente desde la pestaña
Actions. Realiza estos pasos:

1. Instala Python 3.12 y `requirements.txt`.
2. Ejecuta `python main.py`, que descarga el dataset, preprocesa, entrena y evalúa ambos modelos
   y genera el resumen.
3. Publica `resumen_modelos.txt` como artefacto descargable **`resumen-modelos`**.

## 📚 Ubicación de las evidencias

| Evidencia | Ubicación |
|---|---|
| Carga automática y estructura del DataFrame | Notebook, sección 0 · `src/data_loader.py` |
| Preprocesamiento antes/después | Notebook, sección 1 · `src/preprocessing.py` |
| Regresión lineal, métricas y coeficientes | Notebook, sección 2 · `src/regression_model.py` |
| Clasificación, matriz de confusión y errores | Notebook, sección 3 · `src/classification_model.py` |
| Resumen automático | `output/resumen_modelos.txt` · artefacto `resumen-modelos` en Actions |
| Ejecución exitosa | _Completar: enlace a la ejecución de GitHub Actions_ |
