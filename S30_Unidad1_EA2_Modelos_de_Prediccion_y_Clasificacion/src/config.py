"""Configuración del experimento."""

DATASET_ID = "scikit-learn/iris"
DATASET_URL = "https://huggingface.co/datasets/scikit-learn/iris"
DATASET_SPLIT = "train"

COLUMN_MAPPING = {
    "Id": "id",
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species",
}
COLUMNS_TO_DROP = ["id"]
LABEL_COLUMN = "species"
LABEL_PREFIX = "Iris-"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

REGRESSION_TARGET = "petal_length"
REGRESSION_NUMERIC_FEATURES = ["sepal_length", "sepal_width", "petal_width"]
REGRESSION_CATEGORICAL_FEATURES = ["species"]

CLASSIFICATION_TARGET = "species"
CLASSIFICATION_FEATURES = ["sepal_length", "sepal_width",
                           "petal_length", "petal_width"]
LOGISTIC_MAX_ITER = 1000

SUMMARY_FILENAME = "resumen_modelos.txt"
