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

[`manuelgscom/dale-bash-test-v2`](https://huggingface.co/datasets/manuelgscom/dale-bash-test-v2) (split `test`)

471 peticiones en lenguaje natural (español e inglés) dirigidas a un asistente de terminal
Linux. Cada fila incluye la categoría, el entorno (`context`: distribución, gestor de paquetes y
herramientas instaladas), el tipo de respuesta esperado (`command`, `refuse` o `unsure`) y los
comandos bash de referencia. Se descarga directamente del Hub de Hugging Face mediante
`datasets.load_dataset`, sin archivos CSV manuales. El dataset solo publica el split `test`.

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
loader = HFDatasetLoader("manuelgscom/dale-bash-test-v2", split="test")
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
- Frecuencia de valores de las columnas categóricas (top 10 si hay muchas; las listas se
  cuentan como texto y las columnas de texto libre solo indican que todos sus valores son distintos).

```python
summarizer = DatasetSummarizer(df_final)
summarizer.save("output/resumen_dataset.txt")
```

## 🔄 Transformaciones

Todas se aplican en el Notebook, en este orden:

| # | Transformación | Descripción |
|---|----------------|-------------|
| 1 | Expandir columna anidada | `context` (diccionario) → columnas `os`, `distro`, `pm`, `tools` con `pd.json_normalize`. |
| 2 | Seleccionar columnas | Se descartan `os` (siempre `linux`) y `notes` (comentario para evaluadores). |
| 3 | Filtrar registros | Solo peticiones con `expected_type == "command"` (471 → 339 filas). |
| 4 | Columnas derivadas | `request_length` (caracteres de la petición) y `n_tools` (herramientas instaladas). |

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
