"""Limpieza, enriquecimiento, división y construcción de la variable objetivo."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessingStep:
    operation: str
    reason: str
    effect: str


class ListingPreprocessor:
    """Limpia los anuncios y registra cada paso (solo pasos que no aprenden de los datos)."""

    def __init__(self, price_column: str, columns_to_drop: Dict[str, str],
                 categorical: List[str]) -> None:
        self.price_column = price_column
        self.columns_to_drop = columns_to_drop
        self.categorical = categorical
        self.steps: List[PreprocessingStep] = []

    def _log(self, operation: str, reason: str, effect: str) -> None:
        self.steps.append(PreprocessingStep(operation, reason, effect))

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia limpia y enriquecida del DataFrame."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        required = {self.price_column, "multi", "biz", "room_type"}
        missing = required - set(dataframe.columns)
        if missing:
            raise KeyError(f"Faltan columnas en el dataset: {sorted(missing)}")

        self.steps = []
        drop = [c for c in self.columns_to_drop if c in dataframe.columns]
        df = dataframe.drop(columns=drop)
        self._log("Eliminar columnas redundantes",
                  "; ".join(f"{c}: {self.columns_to_drop[c]}" for c in drop),
                  f"{dataframe.shape[1]} -> {df.shape[1]} columnas.")

        both = int(((df["multi"] == 1) & (df["biz"] == 1)).sum())
        df["host_type"] = np.select([df["biz"] == 1, df["multi"] == 1],
                                    ["empresa (>4 anuncios)", "multi (2-4 anuncios)"],
                                    "particular (1 anuncio)")
        df = df.drop(columns=["multi", "biz"])
        self._log("Enriquecer: tipo de anfitrión",
                  "multi y biz son excluyentes (casos con ambas en 1: "
                  f"{both}); juntas describen un solo dato.",
                  f"multi + biz -> host_type: {df['host_type'].value_counts().to_dict()}.")

        rows = len(df)
        nulls = int(df.isnull().sum().sum())
        duplicated = int(df.duplicated().sum())
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        self._log("Nulos y duplicados",
                  "Un anuncio repetido en train y test inflaría las métricas.",
                  f"Nulos: {nulls}; duplicados: {duplicated}; filas: {rows} -> {len(df)}.")

        df["host_is_superhost"] = df["host_is_superhost"].astype(int)
        numeric = [c for c in df.select_dtypes("number").columns if c != self.price_column]
        df[numeric] = df[numeric].astype(float)
        for column in self.categorical:
            df[column] = df[column].astype("category")
        self._log("Corregir tipos",
                  "Los modelos de scikit-learn requieren números; las "
                  "categorías se codifican después dentro del Pipeline.",
                  f"host_is_superhost: bool -> 0/1; numéricas -> float; "
                  f"{self.categorical} -> 'category'.")
        return df

    def steps_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(step) for step in self.steps])


def split_and_build_target(df: pd.DataFrame, features: List[str], price_column: str,
                           test_size: float, random_state: int
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    """Divide y crea precio_alto = precio > mediana de entrenamiento (sin usar prueba)."""
    X_train, X_test, p_train, p_test = train_test_split(
        df[features], df[price_column], test_size=test_size, random_state=random_state)
    median = float(p_train.median())
    y_train = (p_train > median).astype(int).rename("precio_alto")
    y_test = (p_test > median).astype(int).rename("precio_alto")
    return X_train, X_test, y_train, y_test, median
