"""Métricas de clasificación y comparación de algoritmos candidatos."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate(y_true: pd.Series, y_pred: np.ndarray, proba: np.ndarray,
             class_names: Dict[int, str]) -> Dict[str, object]:
    """Métricas en prueba, reporte por clase, matriz de confusión y curva ROC."""
    labels = list(class_names)
    names = [class_names[k] for k in labels]
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                                         index=[f"real_{n}" for n in names],
                                         columns=[f"pred_{n}" for n in names]),
        "report_text": classification_report(y_true, y_pred, labels=labels,
                                             target_names=names, digits=3),
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "n": len(y_true),
    }


def compare_algorithms(X_train: pd.DataFrame, y_train: pd.Series, numeric: List[str],
                       categorical: List[str], folds: int,
                       random_state: int) -> pd.DataFrame:
    """Compara candidatos con la misma CV; escala solo los basados en distancias o pesos."""
    def prep(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])

    candidates = {
        "Random Forest": (RandomForestClassifier(300, random_state=random_state, n_jobs=-1), False),
        "Árbol de decisión (prof. 6)": (DecisionTreeClassifier(max_depth=6, random_state=random_state), False),
        "KNN (k=15)": (KNeighborsClassifier(15), True),
        "Regresión logística": (LogisticRegression(max_iter=2000), True),
        "SVM (kernel RBF)": (SVC(random_state=random_state), True),
        "Naive Bayes gaussiano": (GaussianNB(), True),
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for name, (estimator, scale) in candidates.items():
        scores = cross_validate(Pipeline([("prep", prep(scale)), ("model", estimator)]),
                                X_train[numeric + categorical], y_train, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
        rows.append({"algoritmo": name, "escalado": "sí" if scale else "no",
                     "exactitud_cv": scores["test_accuracy"].mean(),
                     "f1_cv": scores["test_f1"].mean(),
                     "auc_cv": scores["test_roc_auc"].mean()})
    return pd.DataFrame(rows).sort_values("auc_cv", ascending=False).reset_index(drop=True)
