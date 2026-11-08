"""Configuración del experimento."""

BUSINESS_QUESTION = (
    "¿Qué características de un alojamiento de Airbnb en Londres hacen que su "
    "precio de fin de semana quede por encima de la mediana del mercado, para "
    "fijar el precio de un anuncio y decidir en qué mejoras invertir?"
)
BENEFICIARY = ("Anfitriones de Airbnb en Londres y agencias que gestionan "
               "alojamientos por ellos (equipo de precios).")
DECISION = (
    "Ubicar un anuncio en la gama de precio alta o estándar, detectar anuncios "
    "con potencial para cobrar más y priorizar inversiones (capacidad, "
    "habitaciones, calidad del servicio) según su efecto en el precio."
)

DATASET_ID = "kraina/airbnb_london_weekends"
DATASET_URL = "https://huggingface.co/datasets/kraina/airbnb_london_weekends"
DATASET_SPLIT = "train"
DATASET_SOURCE = ("Gyódi, K. y Nawaro, Ł. (2021). Determinants of Airbnb prices "
                  "in European cities: A spatial econometrics approach. "
                  "Tourism Management, 86.")

PRICE_COLUMN = "realSum"
TARGET = "precio_alto"
CLASS_NAMES = {0: "estándar", 1: "alto"}

COLUMNS_TO_DROP = {
    "_id": "identificador del anuncio, no describe el alojamiento",
    "room_shared": "repite room_type (Shared room)",
    "room_private": "repite room_type (Private room)",
    "attr_index_norm": "versión reescalada de attr_index (correlación 1.0)",
    "rest_index_norm": "versión reescalada de rest_index (correlación 1.0)",
}
NUMERIC_FEATURES = ["person_capacity", "bedrooms", "cleanliness_rating",
                    "guest_satisfaction_overall", "dist", "metro_dist",
                    "attr_index", "rest_index", "host_is_superhost"]
CATEGORICAL_FEATURES = ["room_type", "host_type"]
FEATURE_LABELS = {
    "person_capacity": "capacidad (personas)",
    "bedrooms": "habitaciones",
    "cleanliness_rating": "calificación de limpieza (0-10)",
    "guest_satisfaction_overall": "satisfacción de huéspedes (0-100)",
    "dist": "distancia al centro (km)",
    "metro_dist": "distancia al metro (km)",
    "attr_index": "índice de atracciones cercanas",
    "rest_index": "índice de restaurantes cercanos",
    "host_is_superhost": "anfitrión superhost",
    "room_type": "tipo de alojamiento",
    "host_type": "tipo de anfitrión",
}

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
CV_SCORING = "roc_auc"

RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [4, 8, 12, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.5],
}
PERMUTATION_REPEATS = 20
TOP_PARTIAL_DEPENDENCE = 4

OUTPUT_DIR = "output"
FIGURES_DIR = "output/figuras"
SUMMARY_FILENAME = "resumen_ejecucion.txt"
