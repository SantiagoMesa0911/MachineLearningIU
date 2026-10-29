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


def _format_params(params: Dict[str, object]) -> List[str]:
    return [f"    {name} = {value}" for name, value in sorted(params.items())]


class ModelSummaryWriter:
    """Construye y guarda el resumen de una ejecución del experimento."""

    def __init__(self, results: ExperimentResults) -> None:
        self.results = results

    @staticmethod
    def _section(title: str) -> List[str]:
        return ["", SEPARATOR, title, SEPARATOR]

    def _dataset_section(self) -> List[str]:
        r = self.results
        return self._section("1. DATASET") + [
            f"Identificador: {config.DATASET_ID}",
            f"Enlace:        {config.DATASET_URL}",
            f"Split:         {config.DATASET_SPLIT} (único split publicado)",
            "Contenido:     peticiones en lenguaje natural (es/en) a un asistente de "
            "terminal Linux, con el tipo de respuesta esperado y comandos bash de "
            "referencia.",
            "Obtención:     automática con datasets.load_dataset() -> pandas",
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
            "  - StandardScaler: escala las variables numéricas (media 0, desviación 1).",
            "  - OneHotEncoder(drop='first') (regresión): lang y category -> "
            "variables 0/1.",
            "  - TfidfVectorizer (clasificación): convierte el texto de la petición "
            "en pesos de palabras y bigramas; el vocabulario se aprende solo con "
            "entrenamiento.",
            f"División entrenamiento/prueba: test_size={config.TEST_SIZE}, "
            f"random_state={config.RANDOM_STATE}, estratificada.",
        ]
        return lines

    def _regression_section(self) -> List[str]:
        r = self.results
        model, metrics, unit = r.regression, r.regression_metrics, config.REGRESSION_UNIT
        target = r.df_regression[model.target]
        lines = self._section("3. MODELO DE REGRESIÓN LINEAL") + [
            "Objetivo: estimar la longitud (en caracteres) del comando bash que "
            "resuelve una petición, como medida de su complejidad.",
            f"Datos: solo peticiones con {config.REGRESSION_FILTER_COLUMN} = "
            f"'{config.REGRESSION_FILTER_VALUE}' ({len(r.df_regression)} filas); "
            f"las demás no tienen comando.",
            f"Variable objetivo:      {model.target} (media {target.mean():.1f}, "
            f"mín {target.min():.0f}, máx {target.max():.0f} {unit})",
            f"Variables predictoras:  {model.features}",
            "Algoritmo: sklearn.linear_model.LinearRegression (mínimos cuadrados "
            "ordinarios)",
            "Parámetros (todos predeterminados de scikit-learn):",
            *_format_params(model.params()),
            f"Entrenamiento / prueba: {r.regression_sizes['train']} / "
            f"{r.regression_sizes['test']} filas (estratificado por "
            f"{config.REGRESSION_STRATIFY})",
            "",
            "Métricas en prueba:",
            f"    MAE  = {metrics['mae']:.3f} {unit}",
            f"    RMSE = {metrics['rmse']:.3f} {unit}",
            f"    R²   = {metrics['r2']:.4f}",
            f"    MAE modelo base (media) = {metrics['baseline_mae']:.3f} {unit}",
            f"    R² validación cruzada ({config.CV_FOLDS} pliegues) = "
            f"{r.regression_cv['r2_mean']:.4f} ± {r.regression_cv['r2_std']:.4f}",
            "",
            "Coeficientes:",
            model.coefficients().round(3).to_string(index=False),
            "",
            "Interpretación:",
        ]
        lines += [f"  - {text}" for text in interpret_regression(
            model, metrics, r.regression_cv, config.CV_FOLDS, unit,
            config.FEATURE_UNITS)]
        return lines

    def _classification_section(self) -> List[str]:
        r = self.results
        model, metrics = r.classifier, r.classification_metrics
        distribution = r.df_clean[model.target].value_counts().sort_index()
        excluded = [f"    {col}: {why}" for col, why in config.CLASSIFICATION_EXCLUDED.items()]
        lines = self._section("4. MODELO DE CLASIFICACIÓN") + [
            "Objetivo: decidir, a partir del texto de la petición, si el asistente "
            "debe responder con un comando (command), rechazarla por estar fuera "
            "de ámbito (refuse) o pedir aclaración por ambigua (unsure).",
            f"Variable objetivo:      {model.target}  (clases: {model.classes})",
            f"Variables predictoras:  {model.text_feature} (TF-IDF) + "
            f"{model.numeric_features}",
            "Variables excluidas por revelar la respuesta (fuga de información):",
            *excluded,
            "Algoritmo: TfidfVectorizer + sklearn.linear_model.LogisticRegression. "
            "Es un modelo lineal, rápido e interpretable (cada palabra tiene un "
            "peso por clase) y funciona bien con textos cortos y pocos datos.",
            f"TF-IDF: ngram_range={config.TFIDF_NGRAM_RANGE}, "
            f"min_df={config.TFIDF_MIN_DF}, sublinear_tf=True "
            f"(vocabulario aprendido: {model.vocabulary_size()} términos)",
            "Hiperparámetros de LogisticRegression (class_weight, max_iter y "
            "random_state fijados; el resto predeterminados):",
            *_format_params(model.params()),
            "    Nota: en scikit-learn >= 1.8 'penalty' figura como 'deprecated'; "
            "la regularización la define l1_ratio = 0.0, es decir, L2 con "
            "fuerza 1/C.",
            f"Entrenamiento / prueba: {r.classification_sizes['train']} / "
            f"{r.classification_sizes['test']} filas (estratificado por clase)",
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
        lines += [f"  - {text}" for text in interpret_classification(
            model, metrics, r.classification_cv, config.CV_FOLDS)]
        return lines

    def _conclusions_section(self) -> List[str]:
        r = self.results
        reg, clf = r.regression_metrics, r.classification_metrics
        unsure = clf["report"].get("unsure", {})
        return self._section("5. CONCLUSIONES Y LIMITACIONES") + [
            f"  - La regresión estima la longitud del comando con un error medio "
            f"de {reg['mae']:.1f} {config.REGRESSION_UNIT} (R² = {reg['r2']:.3f}). "
            f"La longitud de la petición y el número de rutas ayudan, pero la "
            f"mayor parte de la variación depende de la sintaxis concreta de "
            f"cada comando, que no se observa en la petición.",
            f"  - El objetivo de la regresión es un conteo construido a partir de "
            f"reference_commands; solo se usa como objetivo, nunca como "
            f"predictor. Tiene cola larga (pocos comandos muy largos), lo que "
            f"aumenta el RMSE.",
            f"  - El clasificador alcanza F1 macro = {clf['f1_macro']:.3f}. "
            f"Distingue bien command de refuse por el vocabulario, pero la clase "
            f"unsure (solo {int(unsure.get('support', 0))} casos en prueba) es la "
            f"más difícil: una petición ambigua usa las mismas palabras que una "
            f"petición válida.",
            "  - Sin class_weight='balanced' el modelo tendería a no predecir "
            "nunca la clase minoritaria; el balanceo mejora su recall a costa de "
            "algunos falsos positivos.",
            f"  - Dataset pequeño ({len(r.df_clean)} peticiones) y bilingüe: el "
            f"vocabulario se reparte entre español e inglés. Por eso se reporta "
            f"también validación cruzada.",
            "  - Los modelos son lineales: no capturan el significado de la "
            "petición, solo qué palabras aparecen y algunos rasgos numéricos.",
        ]

    def build(self) -> str:
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
        path = Path(output_path)
        if path.suffix.lower() != ".txt":
            raise ValueError("El archivo de salida debe tener extensión .txt.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(), encoding="utf-8")
        return path
