"""Ejecuta el flujo completo y genera el resumen .txt y las gráficas."""

import shutil
import sys
from pathlib import Path
from typing import List

from src import config, plots
from src.pipeline import ExperimentResults, run_experiment
from src.report_writer import RunSummaryWriter

ROOT = Path(__file__).resolve().parent


def make_figures(results: ExperimentResults) -> List[Path]:
    """Genera las gráficas en output/figuras y devuelve sus rutas relativas."""
    folder = ROOT / config.FIGURES_DIR
    shutil.rmtree(folder, ignore_errors=True)  # evita gráficas de ejecuciones viejas
    labels = config.FEATURE_LABELS
    prices = results.df_clean[config.PRICE_COLUMN]
    paths = [
        plots.price_distribution(prices, results.median_price,
                                 folder / "01_distribucion_precios.png"),
        plots.algorithm_comparison(results.comparison, "Random Forest",
                                   folder / "02_comparacion_algoritmos.png"),
        plots.hyperparameter_search(results.model.grid_results, results.best_params,
                                    folder / "03_busqueda_hiperparametros.png"),
        plots.confusion_and_roc(results.metrics, folder / "04_matriz_confusion_roc.png"),
        plots.feature_importance(results.importance, labels,
                                 folder / "05_importancia_variables.png"),
        plots.partial_dependence(results.dependence, labels,
                                 folder / "06_dependencia_parcial.png"),
        plots.category_effects(results.category_effects, labels,
                               folder / "07_efecto_categorias.png"),
    ]
    return [path.relative_to(ROOT) for path in paths]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Pregunta de negocio: {config.BUSINESS_QUESTION}")
    print(f"Cargando '{config.DATASET_ID}' desde Hugging Face...")
    results = run_experiment()
    figures = make_figures(results)

    path = RunSummaryWriter(results, figures).save(
        ROOT / config.OUTPUT_DIR / config.SUMMARY_FILENAME)
    print(f"Resumen guardado en: {path}\n")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
