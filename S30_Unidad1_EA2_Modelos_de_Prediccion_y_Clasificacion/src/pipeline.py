"""Flujo completo del experimento."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from . import config
from .classification_model import TextLogisticClassifier
from .data_loader import HFDatasetLoader
from .preprocessing import DataPreprocessor, split_data
from .regression_model import LinearRegressionModel


@dataclass
class ExperimentResults:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    df_regression: pd.DataFrame
    preprocessor: DataPreprocessor
    regression: LinearRegressionModel
    regression_metrics: Dict[str, float]
    regression_cv: Dict[str, float]
    regression_sizes: Dict[str, int]
    classifier: TextLogisticClassifier
    classification_metrics: Dict[str, object]
    classification_cv: Dict[str, float]
    classification_sizes: Dict[str, int]


def load_raw_data() -> pd.DataFrame:
    return HFDatasetLoader(config.DATASET_ID, split=config.DATASET_SPLIT).load()


def build_preprocessor() -> DataPreprocessor:
    return DataPreprocessor(config.CONTEXT_COLUMN, config.TEXT_COLUMN,
                            config.COMMANDS_COLUMN, config.TOOLS_COLUMN,
                            config.COLUMNS_TO_DROP, config.CATEGORICAL_COLUMNS)


def regression_subset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Solo las peticiones que se responden con un comando (tienen command_length)."""
    subset = df_clean[df_clean[config.REGRESSION_FILTER_COLUMN]
                      == config.REGRESSION_FILTER_VALUE].reset_index(drop=True)
    for column in subset.select_dtypes("category").columns:
        subset[column] = subset[column].cat.remove_unused_categories()
    return subset


def build_regression() -> LinearRegressionModel:
    return LinearRegressionModel(config.REGRESSION_NUMERIC_FEATURES,
                                 config.REGRESSION_CATEGORICAL_FEATURES,
                                 config.REGRESSION_TARGET)


def build_classifier() -> TextLogisticClassifier:
    return TextLogisticClassifier(
        config.CLASSIFICATION_TEXT_FEATURE, config.CLASSIFICATION_NUMERIC_FEATURES,
        config.CLASSIFICATION_TARGET, ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF, class_weight=config.CLASS_WEIGHT,
        max_iter=config.LOGISTIC_MAX_ITER, random_state=config.RANDOM_STATE)


def run_experiment() -> ExperimentResults:
    """Carga, limpia, entrena y evalúa ambos modelos."""
    df_raw = load_raw_data()
    preprocessor = build_preprocessor()
    df_clean = preprocessor.clean(df_raw)

    df_regression = regression_subset(df_clean)
    regression = build_regression()
    X_train, X_test, y_train, y_test = split_data(
        df_regression, regression.features, config.REGRESSION_TARGET,
        config.TEST_SIZE, config.RANDOM_STATE,
        stratify_column=config.REGRESSION_STRATIFY)
    regression.fit(X_train, y_train)
    regression_metrics = regression.evaluate(X_test, y_test)
    regression_cv = regression.cross_validate(X_train, y_train, config.CV_FOLDS)
    regression_sizes = {"train": len(X_train), "test": len(X_test)}

    classifier = build_classifier()
    X_train, X_test, y_train, y_test = split_data(
        df_clean, classifier.features, config.CLASSIFICATION_TARGET,
        config.TEST_SIZE, config.RANDOM_STATE,
        stratify_column=config.CLASSIFICATION_TARGET)
    classifier.fit(X_train, y_train)
    classification_metrics = classifier.evaluate(X_test, y_test)
    classification_cv = classifier.cross_validate(X_train, y_train,
                                                  config.CV_FOLDS,
                                                  config.RANDOM_STATE)
    classification_sizes = {"train": len(X_train), "test": len(X_test)}

    return ExperimentResults(df_raw, df_clean, df_regression, preprocessor,
                             regression, regression_metrics, regression_cv,
                             regression_sizes, classifier,
                             classification_metrics, classification_cv,
                             classification_sizes)
