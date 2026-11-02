"""Constantes partagées par le service, les scripts et les tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
REFERENCE_PATH = ROOT / "data" / "reference.csv"
TEST_SET_PATH = ROOT / "data" / "test_set.csv"

TARGET = "default"
NUMERIC_FEATURES = ["age", "income", "loan_amount", "employment_years", "num_credit_lines"]
CATEGORICAL_FEATURES = ["region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Contrat des données : utilisé par l'API (validation des requêtes)
# et par les tests de validation des données.
ALLOWED_REGIONS = ["North", "South", "East", "West"]
VALUE_RANGES = {
    "age": (18, 100),
    "income": (0, 1_000_000),
    "loan_amount": (1, 500_000),
    "employment_years": (0, 60),
    "num_credit_lines": (0, 30),
}

# Seuils de validation du modèle (fixés AVANT l'entraînement)
MIN_ROC_AUC = 0.75
MIN_RECALL = 0.60

# Seuil de décision : au-delà, le dossier part en revue humaine
DECISION_THRESHOLD = 0.5
