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
| Identificador | `scikit-learn/iris` |
| Enlace | https://huggingface.co/datasets/scikit-learn/iris |
| Split / configuración | `train` (único split; sin configuraciones) |
| Contenido | 150 flores de 3 especies de iris (*setosa*, *versicolor*, *virginica*), 50 de cada una |
| Columnas | `Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` |

Se descarga con `datasets.load_dataset()` desde el código (`src/data_loader.py`). No hay
archivos manuales.

- **Regresión:** objetivo `petal_length`, que es numérico continuo.
- **Clasificación:** objetivo `species`, que es categórico. Es una etiqueta original del dataset
  y no se derivó de ninguna variable numérica.

## 🗂️ Estructura

```text
S30_Unidad1_EA2_Modelos_de_Prediccion_y_Clasificacion/
├── main.py                          # Punto de entrada del flujo completo (lo usa GitHub Actions)
├── src/
│   ├── config.py                    # Dataset, semilla, test_size, variables de cada modelo
│   ├── data_loader.py               # HFDatasetLoader: Hugging Face -> pandas.DataFrame
│   ├── preprocessing.py             # DataPreprocessor (limpieza) y split_data
│   ├── regression_model.py          # LinearRegressionModel (Pipeline + métricas + coeficientes)
│   ├── classification_model.py      # LogisticRegressionClassifier (Pipeline + matriz + métricas)
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
| Renombrar columnas | Nombres poco cómodos (`SepalLengthCm`) | snake_case: `sepal_length`, ... |
| Eliminar `id` | Solo numera filas; el archivo está ordenado por especie, así que delataría la clase | 6 → 5 columnas |
| Normalizar etiquetas y tipo | Prefijo `Iris-` redundante; la especie era texto | `setosa`, ... con tipo `category` |
| Revisar nulos | scikit-learn no acepta faltantes | 0 nulos |
| Eliminar duplicados | 3 flores repetidas podían quedar en train y test a la vez | 150 → 147 filas |

Las operaciones que **sí aprenden de los datos** van dentro de un `Pipeline` de scikit-learn,
de modo que se ajustan **solo con entrenamiento** y luego se aplican a prueba:

- `StandardScaler` en las variables numéricas, para ambos modelos.
- `OneHotEncoder(drop="first")` en `species`, solo en la regresión. `setosa` queda como referencia.

Para ambos modelos la división es 70 % entrenamiento y 30 % prueba (102 / 45 filas), con
`random_state=42` y estratificada por especie.
El notebook demuestra que las medias del escalador son las de `X_train`.

## 📈 Modelo 1: Regresión lineal

- **Pregunta:** ¿se puede estimar el largo del pétalo a partir de las demás medidas y la especie?
- **Objetivo:** `petal_length` (cm).
- **Predictoras:** `sepal_length`, `sepal_width`, `petal_width` y `species`.
- **Algoritmo:** `LinearRegression`, con todos sus parámetros predeterminados.
- **Métricas:** MAE, RMSE y R² en prueba, comparadas con un modelo base que predice la media, más
  R² con validación cruzada de 5 pliegues.
- **Interpretación:** los coeficientes se reportan por desviación estándar (escala del Pipeline) y
  por cm; el intercepto, en ambas escalas.

## 🎯 Modelo 2: Clasificación

- **Qué se clasifica:** la especie de la flor (3 clases balanceadas).
- **Predictoras:** las 4 medidas.
- **Algoritmo:** `LogisticRegression` multinomial. Es lineal, interpretable y adecuado para pocas
  variables numéricas y clases casi separables.
- **Hiperparámetros:** `max_iter=1000` y `random_state=42`; el resto son predeterminados
  (`C=1.0`, `solver="lbfgs"`, L2).
- **Métricas:** matriz de confusión, exactitud, y precisión, recall y F1 macro, además de
  validación cruzada estratificada de 5 pliegues.

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
