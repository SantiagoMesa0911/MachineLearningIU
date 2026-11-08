"""Random Forest para clasificar anuncios en precio alto o estándar."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class PriceLevelForest:
    """One-hot + RandomForestClassifier. Sin escalado: los árboles usan umbrales."""

    def __init__(self, numeric: List[str], categorical: List[str],
                 random_state: int, params: Optional[Dict[str, object]] = None) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.features = numeric + categorical
        self.random_state = random_state
        self.params = params or {}
        self.pipeline = self._build()
        self.grid_results: Optional[pd.DataFrame] = None

    def _build(self) -> Pipeline:
        encoder = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical)],
            remainder="passthrough", verbose_feature_names_out=False)
        forest = RandomForestClassifier(random_state=self.random_state, n_jobs=-1,
                                        **self.params)
        return Pipeline([("encode", encoder), ("forest", forest)])

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, grid: Dict[str, list],
             folds: int, scoring: str) -> Dict[str, object]:
        """Elige hiperparámetros con validación cruzada sobre entrenamiento."""
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(self._build(), {f"forest__{k}": v for k, v in grid.items()},
                              scoring=scoring, cv=cv, n_jobs=-1)
        search.fit(X_train[self.features], y_train)
        self.params = {k.replace("forest__", ""): v for k, v in search.best_params_.items()}
        self.pipeline = self._build()
        res = pd.DataFrame(search.cv_results_)
        table = {k: [str(p[f"forest__{k}"]) for p in res["params"]] for k in grid}
        self.grid_results = pd.DataFrame({**table, "score_cv": res["mean_test_score"],
                                          "score_cv_std": res["std_test_score"]}
                                         ).sort_values("score_cv", ascending=False)
        return {**self.params, "score_cv": float(search.best_score_)}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PriceLevelForest":
        self.pipeline.fit(X_train[self.features], y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad de precio alto."""
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def permutation_importance(self, X: pd.DataFrame, y: pd.Series, scoring: str,
                               repeats: int) -> pd.DataFrame:
        """Cuánto empeora la métrica al desordenar cada variable (en prueba)."""
        result = permutation_importance(self.pipeline, X[self.features], y,
                                        scoring=scoring, n_repeats=repeats,
                                        random_state=self.random_state, n_jobs=-1)
        return (pd.DataFrame({"variable": self.features,
                              "importancia": result.importances_mean,
                              "desviacion": result.importances_std})
                .sort_values("importancia", ascending=False).reset_index(drop=True))

    def partial_dependence(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Probabilidad media de precio alto al variar una sola variable."""
        result = partial_dependence(self.pipeline, X[self.features], [feature],
                                    kind="average", grid_resolution=30,
                                    percentiles=(0.02, 0.98))
        return pd.DataFrame({"valor": result["grid_values"][0],
                             "probabilidad": result["average"][0]})

    def category_effect(self, X: pd.DataFrame, feature: str) -> pd.Series:
        """Probabilidad media de precio alto fijando cada categoría para todos."""
        effects = {}
        for value in X[feature].cat.categories:
            changed = X[self.features].copy()
            changed[feature] = pd.Categorical([value] * len(changed),
                                              categories=X[feature].cat.categories)
            effects[value] = float(self.predict_proba(changed).mean())
        return pd.Series(effects).sort_values(ascending=False)

    def get_params(self) -> Dict[str, object]:
        return self.pipeline.named_steps["forest"].get_params()
