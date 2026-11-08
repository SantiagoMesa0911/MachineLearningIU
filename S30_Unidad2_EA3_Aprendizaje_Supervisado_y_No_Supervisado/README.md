# S30 - EA3 Unidad 2 - Aprendizaje supervisado y no supervisado

Proyecto que responde una **pregunta de negocio** con un modelo de Machine Learning entrenado sobre
un dataset de Hugging Face. Todo el flujo se ejecuta automáticamente en **GitHub Actions** con cada
push.

```
Pregunta de negocio
        ↓
Hugging Face → DataFrame → preprocesamiento
                         → modelo supervisado (Random Forest)
                         → métricas y evaluación
                         → resumen automático .txt + gráficas
                         → documento de hallazgos (HALLAZGOS.md)
```

## 👥 Integrantes

- Santiago Mesa Parra
- Tatiana Orozco Alzate

## ❓ Pregunta de negocio

> **¿Qué características de un alojamiento de Airbnb en Londres hacen que su precio de fin de
> semana quede por encima de la mediana del mercado, para fijar el precio de un anuncio y decidir
> en qué mejoras invertir?**

| | |
|---|---|
| **Quién la necesita** | Anfitriones de Airbnb en Londres y las agencias que gestionan alojamientos por ellos (equipo de precios) |
| **Qué decisión apoya** | Ubicar un anuncio en la gama alta o estándar, detectar anuncios con potencial para cobrar más y priorizar inversiones (capacidad, habitaciones, calidad del servicio) según su efecto en el precio |
| **Por qué importa** | Un precio por debajo del mercado deja ingresos sin cobrar; uno por encima deja el alojamiento sin reservas |

## 📊 Dataset

| Campo | Valor |
|---|---|
| Identificador | `kraina/airbnb_london_weekends` |
| Enlace | https://huggingface.co/datasets/kraina/airbnb_london_weekends |
| Split / configuración | `train` (único split publicado; configuración `default`) |
| Fuente original | Gyódi, K. y Nawaro, Ł. (2021). *Determinants of Airbnb prices in European cities: A spatial econometrics approach.* Tourism Management, 86 |
| Contenido | 5 379 anuncios reales de Airbnb en Londres para fin de semana, 18 columnas |

**Por qué permite responder la pregunta:** cada anuncio tiene su **precio** y las características
que un anfitrión conoce o puede modificar:

| Grupo | Columnas |
|---|---|
| Precio | `realSum`: precio total de la estancia en €; según el estudio original, para 2 personas y 2 noches |
| Alojamiento | `room_type`, `person_capacity`, `bedrooms` |
| Calidad del servicio | `cleanliness_rating`, `guest_satisfaction_overall`, `host_is_superhost` |
| Anfitrión | `multi` (2–4 anuncios), `biz` (más de 4 anuncios) |
| Ubicación | `dist` (km al centro), `metro_dist` (km al metro), `attr_index` y `rest_index` (cantidad de atracciones y restaurantes cercanos) |

**Variable objetivo construida:**
- **Regla:** `precio_alto` = 1 si `realSum` supera la **mediana de entrenamiento** (€262.94); si no,
  vale 0.
- **Por qué la mediana:** deja dos clases equilibradas y evita que unos pocos precios extremos (el
  máximo es €12 937) dominen el modelo.
- **Sin fuga de información:** la mediana se calcula solo con entrenamiento, y `realSum` no entra
  como predictora.

El dataset se descarga desde el código con `datasets.load_dataset()` (`src/data_loader.py`), sin
archivos manuales.

## 🗂️ Estructura

```text
S30_Unidad2_EA3_Aprendizaje_Supervisado_y_No_Supervisado/
├── main.py                      # Punto de entrada: ejecuta todo y genera resumen y gráficas
├── src/
│   ├── config.py                # Pregunta de negocio, dataset, variables, semilla y rejilla
│   ├── data_loader.py           # HFDatasetLoader: Hugging Face -> DataFrame
│   ├── preprocessing.py         # ListingPreprocessor y split_and_build_target
│   ├── model.py                 # PriceLevelForest (one-hot + Random Forest)
│   ├── evaluation.py            # Métricas y comparación de algoritmos
│   ├── plots.py                 # 7 gráficas PNG
│   ├── pipeline.py              # run_experiment(): orquesta el flujo
│   └── report_writer.py         # RunSummaryWriter: genera el resumen .txt
├── notebooks/
│   └── evidencias_ea3.ipynb     # Evidencias paso a paso (ejecutado)
├── output/
│   ├── resumen_ejecucion.txt    # Resumen generado por la ejecución
│   └── figuras/                 # Gráficas generadas por el código
├── HALLAZGOS.md                 # Documento de hallazgos para el negocio
├── EXPLICACION_CODIGO.md        # Guía de qué hace cada parte del código
├── requirements.txt
└── .gitignore

.github/workflows/s30_ea3_supervisado.yml   # (raíz del repo) workflow de GitHub Actions
```

## 🧹 Preprocesamiento

| Operación | Problema que resuelve | Efecto |
|---|---|---|
| Eliminar columnas redundantes | `_id` es un identificador; `room_shared` y `room_private` repiten `room_type`; los índices `*_norm` son copias reescaladas (correlación 1.0) | 18 → 13 columnas |
| Enriquecer: `host_type` | `multi` y `biz` son excluyentes y describen un solo dato | Una categoría: particular, multi (2–4) o empresa (más de 4) |
| Nulos y duplicados | Un anuncio repetido en train y test inflaría las métricas | 0 nulos y 0 duplicados (se controlan igual) |
| Corregir tipos | scikit-learn requiere números | `host_is_superhost` pasa a 0/1, las numéricas a `float` y las categorías a `category` |
| Objetivo con la mediana de entrenamiento | Precio continuo con valores extremos | Dos clases equilibradas, alrededor de 50 / 50 |
| **One-hot** (dentro del Pipeline) | Random Forest necesita números | `room_type` y `host_type` se convierten en columnas 0/1; aprende solo con entrenamiento |

- **División:** 70 % entrenamiento (3 765) y 30 % prueba (1 614), con `random_state=42`.
- **¿Se escala? No.** Random Forest toma decisiones con umbrales ("¿capacidad > 3?"), así que
  multiplicar una variable por 1000 no cambia ninguna división. En la comparación de algoritmos,
  los métodos **basados en distancias o pesos** (KNN, SVM, regresión logística y Naive Bayes)
  **sí se escalaron** con `StandardScaler`, ajustado solo con entrenamiento.

## 🤖 Modelo: Random Forest

### Tipo de aprendizaje

**Supervisado (clasificación binaria).** La pregunta pide explicar y predecir una variable
**conocida**, la gama de precio de cada anuncio. Un modelo no supervisado agruparía anuncios
parecidos, pero no diría qué los hace caros.

### Cómo funciona, paso a paso

1. **Un árbol de decisión hace preguntas.**
   - Empieza con todos los anuncios de entrenamiento y busca la pregunta que mejor separa los caros
     de los estándar, por ejemplo "¿es una vivienda completa?".
   - Para medir qué tan buena es una pregunta usa la **impureza de Gini**: qué tan mezcladas quedan
     las dos clases en cada lado.
   - Repite el proceso en cada grupo resultante, por ejemplo "¿hay más de 760 restaurantes
     cerca?", hasta llegar a la profundidad máxima o a hojas muy pequeñas.
2. **Un bosque son muchos árboles distintos.** Para que no sean todos iguales, cada árbol:
   - se entrena con una **muestra aleatoria con reemplazo** de los anuncios (*bootstrap*);
   - en cada división solo puede elegir entre un **subconjunto aleatorio de variables**
     (`max_features`).
3. **Los árboles votan.** Para un anuncio nuevo, cada uno de los 300 árboles da su respuesta. La
   **proporción de votos** es la probabilidad de precio alto, y si supera 0.5 el anuncio se
   clasifica como "alto".
4. **Por qué funciona mejor que un solo árbol.** Un árbol solo memoriza fácilmente los datos de
   entrenamiento. Al promediar muchos árboles diferentes, los errores de cada uno se compensan y el
   bosque generaliza mejor.

### Parámetros principales

| Parámetro | Qué significa | Valor elegido | Cómo se eligió |
|---|---|---|---|
| `n_estimators` | Número de árboles que votan | **300** | `GridSearchCV` con [100, 300] |
| `max_depth` | Profundidad máxima de cada árbol: cuántas preguntas seguidas puede hacer | **12** | [4, 8, 12, sin límite]: con 4 se queda corto; sin límite memoriza |
| `min_samples_leaf` | Mínimo de anuncios en cada hoja; evita reglas para casos únicos | **1** | [1, 5, 10] |
| `max_features` | Variables candidatas en cada división; da diversidad entre árboles | **`sqrt`** | [`sqrt`, 50 %] |
| `criterion`, `bootstrap`, `class_weight` | Medida de impureza, muestreo con reemplazo y peso de clases | `gini`, `True`, `None` | Predeterminados (las clases ya están equilibradas) |

- **Cómo se compararon las combinaciones:** con validación cruzada estratificada de **5 pliegues**
  sobre entrenamiento, usando el **AUC ROC**, en una rejilla de 48 combinaciones.
- **Semilla:** `random_state=42` en la división, en los pliegues y en el bosque.

### Por qué Random Forest y no otro

Se compararon seis algoritmos con la misma validación cruzada (`output/figuras/02_comparacion_algoritmos.png`):

| Algoritmo | ¿Escalado? | Exactitud (CV) | AUC (CV) |
|---|---|---|---|
| **Random Forest** | no | **0.873** | **0.937** |
| Regresión logística | sí | 0.859 | 0.927 |
| SVM (kernel RBF) | sí | 0.862 | 0.925 |
| KNN (k = 15) | sí | 0.864 | 0.921 |
| Naive Bayes gaussiano | sí | 0.818 | 0.912 |
| Árbol de decisión (profundidad 6) | no | 0.853 | 0.908 |

### Ventajas

- Es el más preciso de los seis.
- No requiere escalar.
- Capta relaciones no lineales; por ejemplo, el efecto de los restaurantes cercanos se satura.
- Capta interacciones entre variables.
- Es robusto a valores atípicos.
- Permite medir la importancia de cada variable.

### Limitaciones

- Es menos interpretable que un solo árbol: no da una regla simple.
- Cuando dos variables están correlacionadas se reparten la importancia. Por ejemplo,
  `rest_index` y `dist` tienen una correlación de −0.75, así que la distancia al centro parece
  menos importante de lo que es.
- No extrapola fuera del rango visto; por ejemplo, anuncios para más de 6 personas.
- Explica **asociaciones, no causas**: subir la capacidad no garantiza por sí sola un precio más alto.

## 📈 Resultados principales (prueba, 1 614 anuncios)

| Métrica | Valor |
|---|---|
| Exactitud | **87.5 %** (adivinar la clase mayoritaria daría 52.8 %) |
| Precisión de "alto" | 91.2 % |
| Recall de "alto" | 84.4 % |
| AUC ROC | **0.946** |

La interpretación para el negocio está en **[HALLAZGOS.md](HALLAZGOS.md)**.

## 💻 Ejecución local

Desde esta carpeta:

```bash
python -m venv .venv
```

Activar el entorno (Windows: `.venv\Scripts\Activate.ps1`; Linux/macOS: `source .venv/bin/activate`):

```bash
pip install -r requirements.txt
```

Flujo completo, que tarda unos 4 minutos por la búsqueda de hiperparámetros y genera
`output/resumen_ejecucion.txt` y `output/figuras/*.png`:

```bash
python main.py
```

Notebook de evidencias:

```bash
jupyter notebook notebooks/evidencias_ea3.ipynb
```

Se requiere Internet para descargar el dataset la primera vez.

## ⚙️ Ejecución automática (GitHub Actions)

El workflow `.github/workflows/s30_ea3_supervisado.yml` se activa con cada **push** a `main` que
modifique esta carpeta o el propio workflow. También se puede lanzar manualmente. Hace lo siguiente:

1. Instala Python 3.12 y `requirements.txt`.
2. Ejecuta `python main.py`, que descarga el dataset, preprocesa, compara algoritmos, ajusta
   hiperparámetros, entrena, evalúa, interpreta y genera el resumen y las gráficas.
3. Publica el resumen y las gráficas como artefacto descargable **`resumen-ea3`**.

## 📚 Ubicación de las evidencias

| Evidencia | Ubicación |
|---|---|
| Pregunta de negocio y DataFrame | README · notebook, sección 0 |
| Preprocesamiento antes/después | Notebook, sección 1 · `src/preprocessing.py` |
| Justificación y funcionamiento del modelo | README · notebook, sección 2 · gráfica 02 |
| Hiperparámetros, métricas, matriz de confusión e interpretación | Notebook, sección 3 · gráficas 03 a 07 |
| Resumen automático | `output/resumen_ejecucion.txt` · artefacto `resumen-ea3` en Actions |
| Documento de hallazgos | [`HALLAZGOS.md`](HALLAZGOS.md) |
| Explicación del código | [`EXPLICACION_CODIGO.md`](EXPLICACION_CODIGO.md) |
| Ejecución exitosa | [GitHub Actions – ejecución 42522282274](https://github.com/SantiagoMesa0911/MachineLearningIU/actions/runs/42522282274) |
