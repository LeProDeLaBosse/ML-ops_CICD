"""Entraîne le modèle de scoring et l'enregistre dans models/.

Le modèle est DÉJÀ fourni (models/model.joblib) : vous n'avez pas besoin de
lancer ce script pendant le TP. Il est là pour que l'entraînement soit
reproductible et documenté.

Usage (depuis la racine du dépôt) :
    python -m scripts.train
"""
import json
from datetime import date

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

MODEL_VERSION = "1.0.0"


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def main():
    train = pd.read_csv(config.REFERENCE_PATH)
    test = pd.read_csv(config.TEST_SET_PATH)

    model = build_pipeline()
    model.fit(train[config.FEATURES], train[config.TARGET])

    proba = model.predict_proba(test[config.FEATURES])[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(test[config.TARGET], proba), 4),
        "recall": round(
            recall_score(test[config.TARGET], proba >= config.DECISION_THRESHOLD), 4
        ),
    }

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_on": date.today().isoformat(),
        "training_data": "data/reference.csv",
        "scikit_learn_version": sklearn.__version__,
        "test_metrics": metrics,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
