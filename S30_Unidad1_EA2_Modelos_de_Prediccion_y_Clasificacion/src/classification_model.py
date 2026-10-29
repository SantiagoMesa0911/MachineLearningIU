"""Modelo de clasificación de texto con regresión logística."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TextLogisticClassifier:
    """Regresión logística sobre TF-IDF del texto más variables numéricas escaladas."""

    def __init__(self, text_feature: str, numeric_features: List[str], target: str,
                 ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2,
                 class_weight: Optional[str] = "balanced", max_iter: int = 1000,
                 random_state: Optional[int] = None) -> None:
        self.text_feature = text_feature
        self.numeric_features = numeric_features
        self.target = target
        self.features = [text_feature] + numeric_features

        preprocessor = ColumnTransformer(
            [("text", TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                      sublinear_tf=True), text_feature),
             ("num", StandardScaler(), numeric_features)],
            verbose_feature_names_out=False,
        )
        self.pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(class_weight=class_weight,
                                         max_iter=max_iter,
                                         random_state=random_state)),
        ])

    @property
    def classes(self) -> List[str]:
        return [str(c) for c in self.pipeline.named_steps["model"].classes_]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TextLogisticClassifier":
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
                                 y_train.astype(str), cv=cv, scoring="f1_macro")
        return {"f1_macro_mean": float(scores.mean()),
                "f1_macro_std": float(scores.std())}

    def vocabulary_size(self) -> int:
        text = self.pipeline.named_steps["preprocess"].named_transformers_["text"]
        return len(text.vocabulary_)

    def top_features(self, n: int = 8) -> Dict[str, List[str]]:
        """Variables con mayor coeficiente positivo para cada clase."""
        model = self.pipeline.named_steps["model"]
        names = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        return {label: list(pd.Series(coefs, index=names).nlargest(n).index)
                for label, coefs in zip(self.classes, model.coef_)}

    def params(self) -> Dict[str, object]:
        return self.pipeline.named_steps["model"].get_params()
