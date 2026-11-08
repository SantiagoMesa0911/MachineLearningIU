"""Flujo completo del experimento."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from . import config
from .data_loader import HFDatasetLoader
from .evaluation import compare_algorithms, evaluate
from .model import PriceLevelForest
from .preprocessing import ListingPreprocessor, split_and_build_target

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


@dataclass
class ExperimentResults:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    preprocessor: ListingPreprocessor
    median_price: float
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    comparison: pd.DataFrame
    model: PriceLevelForest
    best_params: Dict[str, object]
    metrics: Dict[str, object]
    importance: pd.DataFrame
    dependence: Dict[str, pd.DataFrame]
    category_effects: Dict[str, pd.Series]


def load_raw_data() -> pd.DataFrame:
    return HFDatasetLoader(config.DATASET_ID, split=config.DATASET_SPLIT).load()


def build_preprocessor() -> ListingPreprocessor:
    return ListingPreprocessor(config.PRICE_COLUMN, config.COLUMNS_TO_DROP,
                               config.CATEGORICAL_FEATURES)


def run_experiment() -> ExperimentResults:
    """Carga, limpia, compara, ajusta, entrena, evalúa e interpreta."""
    df_raw = load_raw_data()
    preprocessor = build_preprocessor()
    df_clean = preprocessor.clean(df_raw)
    X_train, X_test, y_train, y_test, median = split_and_build_target(
        df_clean, FEATURES, config.PRICE_COLUMN, config.TEST_SIZE, config.RANDOM_STATE)

    comparison = compare_algorithms(X_train, y_train, config.NUMERIC_FEATURES,
                                    config.CATEGORICAL_FEATURES, config.CV_FOLDS,
                                    config.RANDOM_STATE)

    model = PriceLevelForest(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES,
                             config.RANDOM_STATE)
    best_params = model.tune(X_train, y_train, config.RF_GRID, config.CV_FOLDS,
                             config.CV_SCORING)
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test), model.predict_proba(X_test),
                       config.CLASS_NAMES)

    importance = model.permutation_importance(X_test, y_test, config.CV_SCORING,
                                              config.PERMUTATION_REPEATS)
    top_numeric = [v for v in importance["variable"]
                   if v in config.NUMERIC_FEATURES and v != "host_is_superhost"]
    dependence = {f: model.partial_dependence(X_test, f)
                  for f in top_numeric[:config.TOP_PARTIAL_DEPENDENCE]}
    category_effects = {f: model.category_effect(X_test, f)
                        for f in config.CATEGORICAL_FEATURES}

    return ExperimentResults(df_raw, df_clean, preprocessor, median, X_train, X_test,
                             y_train, y_test, comparison, model, best_params, metrics,
                             importance, dependence, category_effects)
