"""Limpieza del DataFrame y división en entrenamiento y prueba."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessingStep:
    operation: str
    reason: str
    effect: str


class DataPreprocessor:
    """Limpia el dataset y registra cada operación aplicada.

    Solo incluye pasos que no aprenden de los datos; el escalado y la
    codificación van en los Pipeline de cada modelo.
    """

    def __init__(self, column_mapping: Dict[str, str], columns_to_drop: List[str],
                 label_column: str, label_prefix: str = "") -> None:
        self.column_mapping = column_mapping
        self.columns_to_drop = columns_to_drop
        self.label_column = label_column
        self.label_prefix = label_prefix
        self.steps: List[PreprocessingStep] = []

    def _log(self, operation: str, reason: str, effect: str) -> None:
        self.steps.append(PreprocessingStep(operation, reason, effect))

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia limpia del DataFrame."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        missing = set(self.column_mapping) - set(dataframe.columns)
        if missing:
            raise KeyError(f"Faltan columnas en el dataset: {sorted(missing)}")

        self.steps = []
        df = dataframe.rename(columns=self.column_mapping)
        self._log("Renombrar columnas",
                  "Nombres en snake_case, sin la unidad 'Cm'.",
                  f"{list(self.column_mapping)} -> {list(self.column_mapping.values())}")

        df = df.drop(columns=self.columns_to_drop)
        self._log("Eliminar columnas",
                  "'id' solo numera las filas y el archivo está ordenado por "
                  "especie, así que delataría la clase.",
                  f"Columnas eliminadas: {self.columns_to_drop}; quedan {df.shape[1]}.")

        before = sorted(df[self.label_column].unique())
        df[self.label_column] = (df[self.label_column].astype(str)
                                 .str.replace(self.label_prefix, "", regex=False)
                                 .astype("category"))
        self._log("Normalizar etiquetas y corregir tipo",
                  "El prefijo 'Iris-' es redundante y la especie es categórica.",
                  f"{before} -> {list(df[self.label_column].cat.categories)} "
                  f"(tipo 'category').")

        nulls = int(df.isnull().sum().sum())
        rows_before = len(df)
        df = df.dropna()
        self._log("Revisar valores nulos",
                  "scikit-learn no acepta valores faltantes.",
                  f"Nulos encontrados: {nulls}; filas eliminadas: "
                  f"{rows_before - len(df)}.")

        rows_before = len(df)
        duplicated = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        self._log("Eliminar filas duplicadas",
                  "Una fila repetida en train y test a la vez infla las métricas.",
                  f"Duplicados encontrados: {duplicated}; filas: "
                  f"{rows_before} -> {len(df)}.")
        return df

    def steps_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(step) for step in self.steps])


def split_data(dataframe: pd.DataFrame, features: List[str], target: str,
               test_size: float, random_state: int,
               stratify_column: Optional[str] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve X_train, X_test, y_train, y_test."""
    stratify = dataframe[stratify_column] if stratify_column else None
    return train_test_split(dataframe[features], dataframe[target],
                            test_size=test_size, random_state=random_state,
                            stratify=stratify)
