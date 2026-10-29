"""Generación de un resumen en texto a partir de un pandas.DataFrame."""

from pathlib import Path
from typing import Union

import pandas as pd


class DatasetSummarizer:
    """Analiza un DataFrame y exporta un resumen a un archivo .txt.

    Responsabilidad única: describir el DataFrame recibido.
    No descarga datos ni se conecta a Hugging Face.

    Ejemplo:
        >>> summarizer = DatasetSummarizer(df_final)
        >>> summarizer.save("output/resumen_dataset.txt")
    """

    MAX_CATEGORIES = 10

    def __init__(self, dataframe: pd.DataFrame,
                 title: str = "Resumen del dataset") -> None:
        """Inicializa el generador de resumen.

        Args:
            dataframe: DataFrame que se va a resumir.
            title: Título que aparecerá al inicio del resumen.

        Raises:
            TypeError: Si no se recibe un pandas.DataFrame.
            ValueError: Si el DataFrame está vacío.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        if dataframe.empty:
            raise ValueError("El DataFrame está vacío; no hay nada que resumir.")

        self.dataframe = dataframe
        self.title = title

    def build_summary(self) -> str:
        """Construye el texto del resumen.

        Incluye: filas, columnas, nombres y tipos de columna, nulos por
        columna, estadísticas descriptivas de las columnas numéricas y
        conteo de valores de las columnas categóricas.

        Returns:
            El resumen como cadena de texto.
        """
        df = self.dataframe
        separator = "=" * 70
        lines = [
            separator,
            self.title.upper(),
            separator,
            "",
            f"Cantidad de filas:    {df.shape[0]}",
            f"Cantidad de columnas: {df.shape[1]}",
            "",
            "COLUMNAS Y TIPOS DE DATO",
            "-" * 70,
        ]
        for column, dtype in df.dtypes.items():
            lines.append(f"  {column:<20} {dtype}")

        lines += ["", "VALORES NULOS POR COLUMNA", "-" * 70]
        for column, nulls in df.isnull().sum().items():
            lines.append(f"  {column:<20} {nulls}")

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            lines += ["", "ESTADÍSTICAS DESCRIPTIVAS (columnas numéricas)",
                      "-" * 70, numeric_df.describe().round(3).to_string()]

        categorical_df = df.select_dtypes(exclude="number")
        for column in categorical_df.columns:
            # Las listas no son hashables: se cuentan como texto
            values = categorical_df[column].map(
                lambda v: (", ".join(map(str, v)) or "(vacía)")
                if hasattr(v, "__len__") and not isinstance(v, str) else v)
            unique = values.nunique()
            title = f"FRECUENCIA DE VALORES: {column} ({unique} valores distintos)"
            if unique == len(values):
                lines += ["", title, "-" * 70,
                          "  Todos los valores son distintos (texto libre o identificador)."]
                continue
            if unique > self.MAX_CATEGORIES:
                title += f" - top {self.MAX_CATEGORIES}"
            lines += ["", title, "-" * 70,
                      values.value_counts().head(self.MAX_CATEGORIES).to_string()]

        lines += ["", separator]
        return "\n".join(lines)

    def save(self, output_path: Union[str, Path]) -> Path:
        """Guarda el resumen en un archivo .txt (UTF-8).

        Crea la carpeta de destino si no existe.

        Args:
            output_path: Ruta del archivo .txt de salida.

        Returns:
            La ruta del archivo generado.

        Raises:
            ValueError: Si la ruta no termina en .txt.
            OSError: Si el archivo no se puede escribir.
        """
        path = Path(output_path)
        if path.suffix.lower() != ".txt":
            raise ValueError("El archivo de salida debe tener extensión .txt.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_summary(), encoding="utf-8")
        return path
