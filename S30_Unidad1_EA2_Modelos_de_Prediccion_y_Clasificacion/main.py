"""Ejecuta el flujo completo y genera el resumen .txt."""

import sys
from pathlib import Path

from src import config
from src.pipeline import run_experiment
from src.report_writer import ModelSummaryWriter

OUTPUT_PATH = Path(__file__).resolve().parent / "output" / config.SUMMARY_FILENAME


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Cargando '{config.DATASET_ID}' desde Hugging Face...")
    results = run_experiment()
    print(f"DataFrame original: {results.df_raw.shape} | limpio: "
          f"{results.df_clean.shape}")

    path = ModelSummaryWriter(results).save(OUTPUT_PATH)
    print(f"Resumen guardado en: {path}\n")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
