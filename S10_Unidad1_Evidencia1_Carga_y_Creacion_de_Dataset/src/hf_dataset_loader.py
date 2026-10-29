"""Carga de datasets desde Hugging Face como pandas.DataFrame."""

from typing import Optional

import pandas as pd
from datasets import load_dataset


class HFDatasetLoader:
    """Carga un dataset del Hub de Hugging Face y lo devuelve como DataFrame.

    Responsabilidad única: obtener los datos y convertirlos a pandas.
    No transforma los datos ni genera archivos.

    Ejemplo:
        >>> loader = HFDatasetLoader("manuelgscom/dale-bash-test-v2", split="test")
        >>> df = loader.load()
    """

    def __init__(self, dataset_id: str, split: str = "train",
                 config_name: Optional[str] = None) -> None:
        """Inicializa el cargador.

        Args:
            dataset_id: Identificador del dataset en Hugging Face
                (por ejemplo, "manuelgscom/dale-bash-test-v2").
            split: Partición a cargar ("train", "test", ...).
            config_name: Configuración del dataset, si tiene varias.

        Raises:
            ValueError: Si el identificador o el split están vacíos.
        """
        if not dataset_id or not dataset_id.strip():
            raise ValueError("El identificador del dataset no puede estar vacío.")
        if not split or not split.strip():
            raise ValueError("El split no puede estar vacío.")

        self.dataset_id = dataset_id.strip()
        self.split = split.strip()
        self.config_name = config_name

    def load(self) -> pd.DataFrame:
        """Descarga el dataset desde Hugging Face y lo convierte a DataFrame.

        Returns:
            Un pandas.DataFrame con los registros del split solicitado.

        Raises:
            RuntimeError: Si la descarga o la conversión fallan.
        """
        try:
            dataset = load_dataset(self.dataset_id, self.config_name,
                                   split=self.split)
        except Exception as error:
            raise RuntimeError(
                f"No se pudo cargar '{self.dataset_id}' (split='{self.split}') "
                f"desde Hugging Face: {error}"
            ) from error

        dataframe = dataset.to_pandas()
        if not isinstance(dataframe, pd.DataFrame):
            raise RuntimeError("La conversión a pandas.DataFrame no fue posible.")
        return dataframe
