"""Modelo de regresión lineal."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearRegressionModel:
    """Regresión lineal con escalado y one-hot dentro de un Pipeline."""

    def __init__(self, numeric_features: List[str],
                 categorical_features: List[str], target: str) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target = target
        self.features = numeric_features + categorical_features

        preprocessor = ColumnTransformer(
            [("num", StandardScaler(), numeric_features),
             ("cat", OneHotEncoder(drop="first"), categorical_features)],
            verbose_feature_names_out=False,
        )
        self.pipeline = Pipeline([("preprocess", preprocessor),
                                  ("model", LinearRegression())])
        self._y_train_mean = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionModel":
        self.pipeline.fit(X_train[self.features], y_train)
        self._y_train_mean = float(y_train.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.features])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """MAE, RMSE y R² en prueba, más el MAE de predecir siempre la media."""
        predictions = self.predict(X_test)
        baseline = np.full(len(y_test), self._y_train_mean)
        return {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": r2_score(y_test, predictions),
            "baseline_mae": mean_absolute_error(y_test, baseline),
            "target_mean": float(y_test.mean()),
            "n_test": len(y_test),
        }

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series,
                       folds: int) -> Dict[str, float]:
        scores = cross_val_score(self.pipeline, X_train[self.features], y_train,
                                 cv=folds, scoring="r2")
        return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std())}

    def _scaler(self) -> StandardScaler:
        return self.pipeline.named_steps["preprocess"].named_transformers_["num"]

    def coefficients(self) -> pd.DataFrame:
        """Coeficientes por desviación estándar y por cm."""
        model = self.pipeline.named_steps["model"]
        names = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        per_unit = model.coef_.copy()
        per_unit[: len(self.numeric_features)] /= self._scaler().scale_
        return pd.DataFrame({"variable": names,
                             "coef_estandarizado": model.coef_,
                             "coef_por_cm": per_unit})

    def intercepts(self) -> Dict[str, float]:
        model = self.pipeline.named_steps["model"]
        scaler = self._scaler()
        per_cm = model.coef_[: len(self.numeric_features)] / scaler.scale_
        return {
            "intercepto_estandarizado": float(model.intercept_),
            "intercepto_unidades_originales":
                float(model.intercept_ - np.sum(per_cm * scaler.mean_)),
        }

    def reference_category(self) -> str:
        encoder = self.pipeline.named_steps["preprocess"].named_transformers_["cat"]
        return str(encoder.categories_[0][0])

    def params(self) -> Dict[str, object]:
        return self.pipeline.named_steps["model"].get_params()
