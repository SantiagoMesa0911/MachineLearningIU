"""Carga de datasets desde Hugging Face."""

from typing import Optional

import pandas as pd
from datasets import load_dataset


class HFDatasetLoader:
    """Descarga un dataset de Hugging Face y lo devuelve como DataFrame."""

    def __init__(self, dataset_id: str, split: str = "train",
                 config_name: Optional[str] = None) -> None:
        if not dataset_id or not dataset_id.strip():
            raise ValueError("El identificador del dataset no puede estar vacío.")
        if not split or not split.strip():
            raise ValueError("El split no puede estar vacío.")
        self.dataset_id = dataset_id.strip()
        self.split = split.strip()
        self.config_name = config_name

    def load(self) -> pd.DataFrame:
        """Descarga el split indicado y lo convierte a pandas."""
        try:
            dataset = load_dataset(self.dataset_id, self.config_name,
                                   split=self.split)
        except Exception as error:
            raise RuntimeError(
                f"No se pudo cargar '{self.dataset_id}' desde Hugging Face: {error}"
            ) from error
        return dataset.to_pandas()
