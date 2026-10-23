# S10 - Unidad 1 - Evidencia de Aprendizaje 1 - Carga y Creación de Dataset

## 👥 Equipo

- Santiago Mesa Parra
- Tatiana Orozco Alzate

## 🎯 Objetivo

Obtener un dataset desde **Hugging Face**, convertirlo en un `pandas.DataFrame`, aplicarle
cuatro transformaciones y generar un resumen en un archivo `.txt`. El trabajo se organiza en
dos clases de Python con responsabilidades separadas, coordinadas desde un Jupyter Notebook,
y se versiona con Git/GitHub.

## 🛠️ Tecnologías

- 🐍Python 3
- 🐼pandas
- 🤗Hugging Face Datasets (`datasets`)
- 📓Jupyter Notebook

## 📊 Dataset

[`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris)

Es el clásico dataset Iris: 150 flores de tres especies (*setosa*, *versicolor* y *virginica*)
con cuatro medidas en centímetros (largo y ancho del sépalo y del pétalo). Se eligió porque es
pequeño, conocido y permite transformaciones fáciles de explicar. Se descarga directamente del
Hub de Hugging Face mediante `datasets.load_dataset`, sin archivos CSV manuales.

## 📌 Estructura del proyecto

```text
S10_Unidad1_Evidencia1_Carga_y_Creacion_de_Dataset/
├── src/
│   ├── __init__.py               # Expone las dos clases del paquete
│   ├── hf_dataset_loader.py      # Clase HFDatasetLoader
│   └── dataset_summarizer.py     # Clase DatasetSummarizer
├── notebooks/
│   └── actividad_1_dataset.ipynb # Orquestador: carga, transforma y resume
├── output/
│   └── resumen_dataset.txt       # Resumen generado del DataFrame final
├── README.md
├── requirements.txt              # Dependencias del proyecto
└── .gitignore
```

# ⚙️ Componentes del proyecto

## HFDatasetLoader

Archivo: `src/hf_dataset_loader.py`

**Responsabilidad:** cargar un dataset desde Hugging Face y devolverlo como `pandas.DataFrame`.

- Recibe el identificador del dataset (`dataset_id`) y, opcionalmente, el `split`
  (por defecto `"train"`) y la configuración (`config_name`).
- `load()` descarga los datos con `load_dataset(...)` y los convierte con `to_pandas()`.
- Valida los parámetros y envuelve los errores de descarga en un `RuntimeError` con un
  mensaje claro.
- No transforma datos ni genera archivos.

```python
loader = HFDatasetLoader("scikit-learn/iris", split="train")
df = loader.load()
```

## DatasetSummarizer

Archivo: `src/dataset_summarizer.py`

**Responsabilidad:** recibir un `pandas.DataFrame`, analizarlo y exportar un resumen a `.txt`.
No se conecta a Hugging Face ni descarga datos.

El resumen incluye:

- Cantidad de filas y de columnas.
- Nombre y tipo de dato de cada columna.
- Cantidad de valores nulos por columna.
- Estadísticas descriptivas de las columnas numéricas.
- Frecuencia de valores de las columnas categóricas.

```python
summarizer = DatasetSummarizer(df_final)
summarizer.save("output/resumen_dataset.txt")
```

## 🔄 Transformaciones

Todas se aplican en el Notebook, en este orden:

| # | Transformación | Descripción |
|---|----------------|-------------|
| 1 | Renombrar columnas | `SepalLengthCm` → `sepal_length`, etc. Nombres en snake_case, claros y consistentes. |
| 2 | Seleccionar columnas | Se conservan las cuatro medidas y `species`; se descarta `id`, que solo numera las filas. |
| 3 | Filtrar registros | Se conservan las flores con `sepal_length >= 5.0` cm (150 → 128 filas). |
| 4 | Columna derivada | `petal_area = petal_length * petal_width`: aproximación del área del pétalo en cm². |

El resultado es `df_final`, que es el DataFrame que se envía a `DatasetSummarizer`.

## 🚀 Ejecución

Desde la carpeta de la actividad:

```bash
python -m venv .venv
```

Activar el entorno (Windows PowerShell: `.venv\Scripts\Activate.ps1`; Linux/macOS:
`source .venv/bin/activate`) e instalar dependencias:

```bash
pip install -r requirements.txt
```

Abrir el Notebook y ejecutar todas las celdas (*Run All*):

```bash
jupyter notebook notebooks/actividad_1_dataset.ipynb
```

O ejecutarlo completo desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/actividad_1_dataset.ipynb
```

La primera ejecución requiere conexión a Internet para descargar el dataset.

## 📄 Resultado

Al ejecutar el Notebook se genera:

```text
output/resumen_dataset.txt
```

con el resumen del **DataFrame final** (después de las cuatro transformaciones).
