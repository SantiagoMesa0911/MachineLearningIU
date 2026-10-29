"""Configuración del experimento."""

DATASET_ID = "manuelgscom/dale-bash-test-v2"
DATASET_URL = "https://huggingface.co/datasets/manuelgscom/dale-bash-test-v2"
DATASET_SPLIT = "test"

CONTEXT_COLUMN = "context"
TEXT_COLUMN = "request"
COMMANDS_COLUMN = "reference_commands"
TOOLS_COLUMN = "tools"
COLUMNS_TO_DROP = {
    "id": "Identificador único: no describe la petición.",
    "os": "Vale 'linux' en todas las filas: no aporta información.",
    "notes": "Comentario para evaluadores que anticipa la respuesta esperada (fuga).",
}
CATEGORICAL_COLUMNS = ["lang", "category", "distro", "pm", "expected_type"]

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

REGRESSION_FILTER_COLUMN = "expected_type"
REGRESSION_FILTER_VALUE = "command"
REGRESSION_TARGET = "command_length"
REGRESSION_UNIT = "caracteres"
REGRESSION_NUMERIC_FEATURES = ["request_length", "n_paths", "n_tools"]
FEATURE_UNITS = {"request_length": "carácter de la petición",
                 "n_paths": "ruta mencionada",
                 "n_tools": "herramienta instalada"}
REGRESSION_CATEGORICAL_FEATURES = ["lang", "category"]
REGRESSION_STRATIFY = "category"

CLASSIFICATION_TARGET = "expected_type"
CLASSIFICATION_TEXT_FEATURE = "request"
CLASSIFICATION_NUMERIC_FEATURES = ["request_length", "n_paths"]
CLASSIFICATION_EXCLUDED = {
    "category": "'fuera_de_ambito' equivale siempre a 'refuse' y 'ambigua' a 'unsure'.",
    "reference_commands": "Está vacía justo cuando no se espera un comando.",
    "command_length": "Derivada de reference_commands.",
}
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
CLASS_WEIGHT = "balanced"
LOGISTIC_MAX_ITER = 1000

SUMMARY_FILENAME = "resumen_modelos.txt"
