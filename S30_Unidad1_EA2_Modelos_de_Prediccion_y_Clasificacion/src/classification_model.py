"""Modelo de clasificación con regresión logística."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionClassifier:
    """Regresión logística multiclase con escalado dentro de un Pipeline."""

    def __init__(self, features: List[str], target: str,
                 max_iter: int = 1000, random_state: Optional[int] = None) -> None:
        self.features = features
        self.target = target
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=max_iter,
                                         random_state=random_state)),
        ])

    @property
    def classes(self) -> List[str]:
        return [str(c) for c in self.pipeline.named_steps["model"].classes_]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionClassifier":
        self.pipeline.fit(X_train[self.features], y_train.astype(str))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipeline.predict(X[self.features]), index=X.index)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
        """Métricas macro, reporte por clase y matriz de confusión."""
        y_true = y_test.astype(str)
        y_pred = self.predict(X_test)
        labels = self.classes
        matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                              index=[f"real_{c}" for c in labels],
                              columns=[f"pred_{c}" for c in labels])
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro",
                                               zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro",
                                         zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "report": classification_report(y_true, y_pred, labels=labels,
                                            output_dict=True, zero_division=0),
            "report_text": classification_report(y_true, y_pred, labels=labels,
                                                 digits=3, zero_division=0),
            "confusion_matrix": matrix,
            "n_test": len(y_true),
        }

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series,
                       folds: int, random_state: int) -> Dict[str, float]:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(self.pipeline, X_train[self.features],
                                 y_train.astype(str), cv=cv, scoring="accuracy")
        return {"accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std())}

    def coefficients(self) -> pd.DataFrame:
        model = self.pipeline.named_steps["model"]
        return pd.DataFrame(model.coef_, index=self.classes, columns=self.features)

    def params(self) -> Dict[str, object]:
        return self.pipeline.named_steps["model"].get_params()
