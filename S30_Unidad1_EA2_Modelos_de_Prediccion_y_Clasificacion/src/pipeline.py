"""Flujo completo del experimento."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from . import config
from .classification_model import LogisticRegressionClassifier
from .data_loader import HFDatasetLoader
from .preprocessing import DataPreprocessor, split_data
from .regression_model import LinearRegressionModel


@dataclass
class ExperimentResults:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    preprocessor: DataPreprocessor
    regression: LinearRegressionModel
    regression_metrics: Dict[str, float]
    regression_cv: Dict[str, float]
    regression_sizes: Dict[str, int]
    classifier: LogisticRegressionClassifier
    classification_metrics: Dict[str, object]
    classification_cv: Dict[str, float]
    classification_sizes: Dict[str, int]


def load_raw_data() -> pd.DataFrame:
    return HFDatasetLoader(config.DATASET_ID, split=config.DATASET_SPLIT).load()


def build_preprocessor() -> DataPreprocessor:
    return DataPreprocessor(config.COLUMN_MAPPING, config.COLUMNS_TO_DROP,
                            config.LABEL_COLUMN, config.LABEL_PREFIX)


def run_experiment() -> ExperimentResults:
    """Carga, limpia, entrena y evalúa ambos modelos."""
    df_raw = load_raw_data()
    preprocessor = build_preprocessor()
    df_clean = preprocessor.clean(df_raw)

    # Regresión lineal
    regression = LinearRegressionModel(config.REGRESSION_NUMERIC_FEATURES,
                                       config.REGRESSION_CATEGORICAL_FEATURES,
                                       config.REGRESSION_TARGET)
    X_train, X_test, y_train, y_test = split_data(
        df_clean, regression.features, config.REGRESSION_TARGET,
        config.TEST_SIZE, config.RANDOM_STATE, stratify_column=config.LABEL_COLUMN)
    regression.fit(X_train, y_train)
    regression_metrics = regression.evaluate(X_test, y_test)
    regression_cv = regression.cross_validate(X_train, y_train, config.CV_FOLDS)
    regression_sizes = {"train": len(X_train), "test": len(X_test)}

    # Clasificación
    classifier = LogisticRegressionClassifier(
        config.CLASSIFICATION_FEATURES, config.CLASSIFICATION_TARGET,
        max_iter=config.LOGISTIC_MAX_ITER, random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = split_data(
        df_clean, config.CLASSIFICATION_FEATURES, config.CLASSIFICATION_TARGET,
        config.TEST_SIZE, config.RANDOM_STATE,
        stratify_column=config.CLASSIFICATION_TARGET)
    classifier.fit(X_train, y_train)
    classification_metrics = classifier.evaluate(X_test, y_test)
    classification_cv = classifier.cross_validate(X_train, y_train,
                                                  config.CV_FOLDS,
                                                  config.RANDOM_STATE)
    classification_sizes = {"train": len(X_train), "test": len(X_test)}

    return ExperimentResults(df_raw, df_clean, preprocessor,
                             regression, regression_metrics, regression_cv,
                             regression_sizes, classifier,
                             classification_metrics, classification_cv,
                             classification_sizes)
