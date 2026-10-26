"""Generación del resumen .txt con los resultados de ambos modelos."""

import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import datasets
import pandas as pd
import sklearn

from . import config
from .interpretation import (describe_class_balance, interpret_classification,
                             interpret_regression)
from .pipeline import ExperimentResults

SEPARATOR = "=" * 78
SUBSEPARATOR = "-" * 78


def _format_params(params: Dict[str, object]) -> List[str]:
    return [f"    {name} = {value}" for name, value in sorted(params.items())]


class ModelSummaryWriter:
    """Construye y guarda el resumen de una ejecución del experimento."""

    def __init__(self, results: ExperimentResults) -> None:
        self.results = results

    def _section(self, title: str) -> List[str]:
        return ["", SEPARATOR, title, SEPARATOR]

    def _dataset_section(self) -> List[str]:
        r = self.results
        return self._section("1. DATASET") + [
            f"Identificador: {config.DATASET_ID}",
            f"Enlace:        {config.DATASET_URL}",
            f"Split:         {config.DATASET_SPLIT} (único split disponible)",
            f"Obtención:     automática con datasets.load_dataset() -> pandas",
            f"Tamaño original: {r.df_raw.shape[0]} filas x {r.df_raw.shape[1]} columnas",
            f"Columnas originales: {list(r.df_raw.columns)}",
            f"Tamaño limpio:   {r.df_clean.shape[0]} filas x {r.df_clean.shape[1]} columnas",
            f"Columnas limpias:    {list(r.df_clean.columns)}",
        ]

    def _preprocessing_section(self) -> List[str]:
        lines = self._section("2. PREPROCESAMIENTO")
        lines.append("Operaciones sobre el DataFrame completo (no aprenden de los datos):")
        for i, step in enumerate(self.results.preprocessor.steps, start=1):
            lines += [f"  {i}. {step.operation}",
                      f"     Motivo: {step.reason}",
                      f"     Efecto: {step.effect}"]
        lines += [
            "Operaciones dentro de cada Pipeline (ajustadas SOLO con entrenamiento):",
            "  - StandardScaler: centra en 0 y escala a desviación 1 las variables "
            "numéricas.",
            "  - OneHotEncoder(drop='first') (solo regresión): convierte 'species' "
            "en variables 0/1.",
            f"División entrenamiento/prueba: test_size={config.TEST_SIZE}, "
            f"random_state={config.RANDOM_STATE}, estratificada por especie.",
        ]
        return lines

    def _regression_section(self) -> List[str]:
        r = self.results
        model, metrics = r.regression, r.regression_metrics
        lines = self._section("3. MODELO DE REGRESIÓN LINEAL") + [
            "Objetivo: estimar el largo del pétalo (cm) a partir de las demás "
            "medidas de la flor y su especie.",
            f"Variable objetivo:      {model.target}",
            f"Variables predictoras:  {model.features}",
            "Algoritmo: sklearn.linear_model.LinearRegression (mínimos cuadrados "
            "ordinarios)",
            "Parámetros (todos predeterminados de scikit-learn):",
            *_format_params(model.params()),
            f"Entrenamiento / prueba: {r.regression_sizes['train']} / "
            f"{r.regression_sizes['test']} filas",
            "",
            "Métricas en prueba:",
            f"    MAE  = {metrics['mae']:.4f} cm",
            f"    RMSE = {metrics['rmse']:.4f} cm",
            f"    R²   = {metrics['r2']:.4f}",
            f"    MAE modelo base (media) = {metrics['baseline_mae']:.4f} cm",
            f"    R² validación cruzada ({config.CV_FOLDS} pliegues) = "
            f"{r.regression_cv['r2_mean']:.4f} ± {r.regression_cv['r2_std']:.4f}",
            "",
            "Coeficientes:",
            model.coefficients().round(4).to_string(index=False),
            "",
            "Interpretación:",
        ]
        lines += [f"  - {text}" for text in
                  interpret_regression(model, metrics, r.regression_cv,
                                       config.CV_FOLDS)]
        return lines

    def _classification_section(self) -> List[str]:
        r = self.results
        model, metrics = r.classifier, r.classification_metrics
        distribution = r.df_clean[model.target].value_counts().sort_index()
        lines = self._section("4. MODELO DE CLASIFICACIÓN") + [
            "Objetivo: identificar la especie de una flor a partir de sus cuatro "
            "medidas.",
            f"Variable objetivo:      {model.target}  (clases: {model.classes})",
            f"Variables predictoras:  {model.features}",
            "Algoritmo: sklearn.linear_model.LogisticRegression (multinomial). "
            "Modelo lineal, interpretable y adecuado para pocas variables "
            "numéricas y clases casi separables.",
            "Hiperparámetros (max_iter y random_state fijados; el resto "
            "predeterminados):",
            *_format_params(model.params()),
            "    Nota: en scikit-learn >= 1.8 'penalty' figura como 'deprecated'; "
            "la regularización la define l1_ratio = 0.0, es decir, L2 con "
            "fuerza 1/C.",
            f"Entrenamiento / prueba: {r.classification_sizes['train']} / "
            f"{r.classification_sizes['test']} filas",
            describe_class_balance(distribution),
            "",
            "Métricas en prueba:",
            f"    Exactitud        = {metrics['accuracy']:.4f}",
            f"    Precisión macro  = {metrics['precision_macro']:.4f}",
            f"    Recall macro     = {metrics['recall_macro']:.4f}",
            f"    F1 macro         = {metrics['f1_macro']:.4f}",
            "",
            "Matriz de confusión (filas = clase real, columnas = predicción):",
            metrics["confusion_matrix"].to_string(),
            "",
            "Reporte por clase:",
            metrics["report_text"].rstrip(),
            "",
            "Interpretación:",
        ]
        lines += [f"  - {text}" for text in
                  interpret_classification(model, metrics, r.classification_cv,
                                           config.CV_FOLDS)]
        return lines

    def _conclusions_section(self) -> List[str]:
        r = self.results
        return self._section("5. CONCLUSIONES Y LIMITACIONES") + [
            f"  - La regresión lineal estima {r.regression.target} con un error "
            f"medio de {r.regression_metrics['mae']:.2f} cm (R² = "
            f"{r.regression_metrics['r2']:.3f}).",
            f"  - El clasificador identifica la especie con exactitud de "
            f"{r.classification_metrics['accuracy']:.3f} y F1 macro de "
            f"{r.classification_metrics['f1_macro']:.3f}.",
            f"  - Dataset pequeño ({len(r.df_clean)} flores; "
            f"{r.classification_metrics['n_test']} en prueba): cada error en "
            f"prueba cambia la exactitud en "
            f"{1 / r.classification_metrics['n_test'] * 100:.1f} puntos. Por eso "
            f"se reporta también validación cruzada.",
            "  - Las medidas están muy correlacionadas entre sí y con la especie "
            "(multicolinealidad): los coeficientes individuales de la regresión "
            "deben leerse con cautela y no implican causalidad.",
            "  - La regresión usa la especie como predictora, así que requiere "
            "conocerla de antemano.",
            "  - La regresión supone relaciones lineales; no captura curvaturas "
            "ni interacciones entre variables.",
            "  - versicolor y virginica se solapan en sus medidas; es ahí donde "
            "el clasificador puede equivocarse.",
            "  - Iris es un conjunto limpio y bien separado; los resultados no se "
            "generalizan a otras especies o condiciones de medición.",
        ]

    def build(self) -> str:
        """Construye el texto completo del resumen."""
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S (UTC%z)")
        header = [
            SEPARATOR,
            "S30 - EA2 UNIDAD 1 - MODELOS DE PREDICCIÓN Y CLASIFICACIÓN",
            "Resumen generado automáticamente por código",
            SEPARATOR,
            f"Fecha de ejecución: {now}",
            f"Entorno: Python {platform.python_version()} | pandas "
            f"{pd.__version__} | scikit-learn {sklearn.__version__} | datasets "
            f"{datasets.__version__}",
        ]
        sections = (header + self._dataset_section()
                    + self._preprocessing_section() + self._regression_section()
                    + self._classification_section()
                    + self._conclusions_section() + ["", SEPARATOR])
        return "\n".join(sections) + "\n"

    def save(self, output_path: Union[str, Path]) -> Path:
        """Guarda el resumen en un archivo .txt (UTF-8)."""
        path = Path(output_path)
        if path.suffix.lower() != ".txt":
            raise ValueError("El archivo de salida debe tener extensión .txt.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(), encoding="utf-8")
        return path
