"""Limpieza y enriquecimiento del DataFrame, y división en entrenamiento y prueba."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Rutas como /srv/fotos, ./script.sh o ~/notas.txt
PATH_PATTERN = r"(?:^|\s)[~.]?/\S+"


@dataclass
class PreprocessingStep:
    operation: str
    reason: str
    effect: str


class DataPreprocessor:
    """Limpia y enriquece el dataset, registrando cada operación.

    Solo incluye pasos que no aprenden de los datos; el escalado, la
    codificación y TF-IDF van en los Pipeline de cada modelo.
    """

    def __init__(self, context_column: str, text_column: str, commands_column: str,
                 tools_column: str, columns_to_drop: Dict[str, str],
                 categorical_columns: List[str]) -> None:
        self.context_column = context_column
        self.text_column = text_column
        self.commands_column = commands_column
        self.tools_column = tools_column
        self.columns_to_drop = columns_to_drop
        self.categorical_columns = categorical_columns
        self.steps: List[PreprocessingStep] = []

    def _log(self, operation: str, reason: str, effect: str) -> None:
        self.steps.append(PreprocessingStep(operation, reason, effect))

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia limpia y enriquecida del DataFrame."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        required = {self.context_column, self.text_column, self.commands_column}
        missing = required - set(dataframe.columns)
        if missing:
            raise KeyError(f"Faltan columnas en el dataset: {sorted(missing)}")

        self.steps = []
        context = pd.json_normalize(dataframe[self.context_column].tolist())
        df = pd.concat([dataframe.drop(columns=self.context_column), context], axis=1)
        self._log("Expandir columna anidada",
                  f"'{self.context_column}' es un diccionario; los modelos necesitan "
                  f"columnas planas.",
                  f"{self.context_column} -> {list(context.columns)}")

        drop = [c for c in self.columns_to_drop if c in df.columns]
        df = df.drop(columns=drop)
        reasons = " ".join(f"{c}: {self.columns_to_drop[c]}" for c in drop)
        self._log("Eliminar columnas", reasons, f"Columnas eliminadas: {drop}.")

        df[self.text_column] = df[self.text_column].str.strip()
        rows_before = len(df)
        nulls = int(df[[self.text_column]].isnull().sum().sum())
        duplicated = int(df[self.text_column].duplicated().sum())
        df = (df.dropna(subset=[self.text_column])
                .drop_duplicates(subset=[self.text_column])
                .reset_index(drop=True))
        self._log("Nulos y duplicados",
                  "Una petición vacía no se puede modelar y una repetida en "
                  "train y test infla las métricas.",
                  f"Nulos: {nulls}; peticiones duplicadas: {duplicated}; filas: "
                  f"{rows_before} -> {len(df)}.")

        for column in self.categorical_columns:
            df[column] = df[column].astype("category")
        self._log("Corregir tipos",
                  "Estas columnas toman pocos valores fijos: son categóricas.",
                  f"{self.categorical_columns} -> tipo 'category'.")

        text = df[self.text_column]
        df["request_length"] = text.str.len()
        df["n_paths"] = text.str.count(PATH_PATTERN)
        df["n_tools"] = df[self.tools_column].apply(len)
        df["command_length"] = df[self.commands_column].apply(
            lambda cmds: float(len(cmds[0])) if len(cmds) else np.nan)
        self._log("Enriquecer con variables numéricas",
                  "El texto y las listas no son numéricos; se resumen en "
                  "variables que los modelos pueden usar.",
                  "request_length (caracteres de la petición), n_paths (rutas "
                  "mencionadas), n_tools (herramientas instaladas), "
                  "command_length (caracteres del comando de referencia; vacío "
                  "si no se espera comando).")

        df = df.drop(columns=[self.tools_column, self.commands_column])
        self._log("Eliminar listas ya resumidas",
                  "Las listas originales no son utilizables por los modelos.",
                  f"Columnas eliminadas: {[self.tools_column, self.commands_column]}; "
                  f"quedan {df.shape[1]}.")
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
