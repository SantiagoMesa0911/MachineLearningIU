"""Generación del resumen .txt de la ejecución."""

import platform
from datetime import datetime
from pathlib import Path
from typing import List, Union

import datasets
import pandas as pd
import sklearn

from . import config
from .pipeline import ExperimentResults

SEPARATOR = "=" * 78


def _section(title: str) -> List[str]:
    return ["", SEPARATOR, title, SEPARATOR]


def _indent(text: str, spaces: int = 6) -> List[str]:
    return [" " * spaces + line for line in text.splitlines()]


class RunSummaryWriter:
    """Construye y guarda el resumen de una ejecución."""

    def __init__(self, results: ExperimentResults, figures: List[Path]) -> None:
        self.results = results
        self.figures = figures

    def build(self) -> str:
        r = self.results
        m, best = r.metrics, r.best_params
        labels = config.FEATURE_LABELS
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S (UTC%z)")
        prices = r.df_clean[config.PRICE_COLUMN]
        forest_params = {k: v for k, v in r.model.get_params().items()
                         if k in ("n_estimators", "max_depth", "min_samples_leaf",
                                  "max_features", "criterion", "bootstrap",
                                  "class_weight", "random_state")}

        lines = [
            SEPARATOR,
            "S30 - EA3 UNIDAD 2 - APRENDIZAJE SUPERVISADO Y NO SUPERVISADO",
            "Resumen generado automáticamente por código",
            SEPARATOR,
            f"Fecha de ejecución: {now}",
            f"Entorno: Python {platform.python_version()} | pandas {pd.__version__} | "
            f"scikit-learn {sklearn.__version__} | datasets {datasets.__version__}",
            *_section("1. PREGUNTA DE NEGOCIO"),
            f"Pregunta:  {config.BUSINESS_QUESTION}",
            f"Quién:     {config.BENEFICIARY}",
            f"Decisión:  {config.DECISION}",
            *_section("2. DATASET"),
            f"Identificador: {config.DATASET_ID}",
            f"Enlace:        {config.DATASET_URL}",
            f"Split:         {config.DATASET_SPLIT} (único publicado)",
            f"Fuente original: {config.DATASET_SOURCE}",
            "Obtención:     automática con datasets.load_dataset() -> pandas",
            f"Tamaño original: {r.df_raw.shape[0]} filas x {r.df_raw.shape[1]} columnas",
            f"Columnas originales: {list(r.df_raw.columns)}",
            f"Tamaño limpio:   {r.df_clean.shape[0]} filas x {r.df_clean.shape[1]} columnas",
            f"Precio ({config.PRICE_COLUMN}, € por estancia): mediana {prices.median():.1f}, "
            f"media {prices.mean():.1f}, mín {prices.min():.1f}, máx {prices.max():.1f}",
            *_section("3. PREPROCESAMIENTO"),
        ]
        for i, step in enumerate(r.preprocessor.steps, start=1):
            lines += [f"  {i}. {step.operation}", f"     Motivo: {step.reason}",
                      f"     Efecto: {step.effect}"]
        lines += [
            f"  5. Variable objetivo: {config.TARGET} = 1 si {config.PRICE_COLUMN} > "
            f"mediana de ENTRENAMIENTO (€{r.median_price:.2f}); si no, 0. La mediana no "
            f"usa datos de prueba. {config.PRICE_COLUMN} se excluye de las predictoras.",
            f"     Balance en entrenamiento: {r.y_train.mean():.1%} alto | prueba: "
            f"{r.y_test.mean():.1%} alto.",
            "  6. Dentro del Pipeline (ajustado SOLO con entrenamiento): OneHotEncoder "
            "para room_type y host_type.",
            "     Sin escalado: Random Forest divide por umbrales, no usa distancias. "
            "En la comparación sí se escalaron KNN, SVM, regresión logística y Naive Bayes.",
            f"  División: {len(r.X_train)} entrenamiento / {len(r.X_test)} prueba "
            f"(test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE}).",
            *_section("4. MODELO"),
            "Tipo de aprendizaje: SUPERVISADO (clasificación binaria): se conoce el "
            "precio de cada anuncio y se predice su gama.",
            "Algoritmo: Random Forest (sklearn RandomForestClassifier).",
            f"Variable objetivo: {config.TARGET} (0 = estándar, 1 = alto)",
            "Variables predictoras: " + ", ".join(
                f"{f} ({labels[f]})" for f in config.NUMERIC_FEATURES
                + config.CATEGORICAL_FEATURES),
            "Comparación de algoritmos (validación cruzada estratificada de "
            f"{config.CV_FOLDS} pliegues sobre entrenamiento):",
            *_indent(r.comparison.round(4).to_string(index=False)),
            f"Búsqueda de hiperparámetros: GridSearchCV, criterio {config.CV_SCORING}, "
            f"rejilla {config.RF_GRID}",
            f"    Mejor combinación: {({k: v for k, v in best.items() if k != 'score_cv'})} "
            f"-> AUC CV = {best['score_cv']:.4f}",
            "    10 mejores combinaciones:",
            *_indent(r.model.grid_results.head(10).round(4).to_string(index=False)),
            f"Parámetros finales del bosque (los no listados en la rejilla son "
            f"predeterminados): {forest_params}",
            *_section("5. MÉTRICAS EN PRUEBA"),
            f"    Exactitud = {m['accuracy']:.4f}  (referencia: adivinar la clase "
            f"mayoritaria = {m['baseline_accuracy']:.4f})",
            f"    Precisión (alto) = {m['precision']:.4f}",
            f"    Recall (alto)    = {m['recall']:.4f}",
            f"    F1 (alto)        = {m['f1']:.4f}",
            f"    AUC ROC          = {m['roc_auc']:.4f}",
            "    Matriz de confusión (filas = real, columnas = predicción):",
            *_indent(m["confusion_matrix"].to_string()),
            "    Reporte por clase:",
            *_indent(m["report_text"].rstrip()),
            *_section("6. INTERPRETACIÓN DEL MODELO"),
            f"Importancia por permutación en prueba (caída del AUC, "
            f"{config.PERMUTATION_REPEATS} repeticiones):",
            *(f"    {row.variable:<28} {row.importancia:.4f} ± {row.desviacion:.4f}"
              for row in r.importance.itertuples()),
            "Dependencia parcial (P(precio alto) en el rango 2-98 % de cada variable):",
            *(f"    {labels[f]}: {d['probabilidad'].iloc[0]:.2f} con {d['valor'].iloc[0]:.2f} "
              f"-> {d['probabilidad'].iloc[-1]:.2f} con {d['valor'].iloc[-1]:.2f}"
              for f, d in r.dependence.items()),
            "Efecto de las categorías (P(precio alto) promedio si todos los anuncios "
            "tuvieran esa categoría):",
            *(f"    {labels[f]}: " + ", ".join(f"{k} {v:.0%}" for k, v in s.items())
              for f, s in r.category_effects.items()),
            *_section("7. GRÁFICAS GENERADAS"),
            *(f"    {path.as_posix()}" for path in self.figures),
            "", SEPARATOR,
        ]
        return "\n".join(lines) + "\n"

    def save(self, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        if path.suffix.lower() != ".txt":
            raise ValueError("El archivo de salida debe tener extensión .txt.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(), encoding="utf-8")
        return path
