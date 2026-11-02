"""Tests de validation du modèle — À COMPLÉTER.

Ces tests chargent le modèle livré (models/model.joblib) et l'évaluent sur le
jeu de test figé (data/test_set.csv). Ils doivent échouer si quelqu'un
commite un modèle moins bon que les seuils définis dans src/config.py.
"""
import joblib
import pandas as pd
import pytest
from sklearn.metrics import recall_score, roc_auc_score

from src import config


@pytest.fixture(scope="module")
def model():
    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def test_set():
    df = pd.read_csv(config.TEST_SET_PATH)
    return df[config.FEATURES], df[config.TARGET]


def test_roc_auc_above_threshold(model, test_set):
    # ROC AUC sur le jeu de test >= config.MIN_ROC_AUC
    X, y = test_set
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) >= config.MIN_ROC_AUC


def test_recall_above_threshold(model, test_set):
    # rappel (classe 1, seuil config.DECISION_THRESHOLD) >= config.MIN_RECALL
    X, y = test_set
    predictions = model.predict_proba(X)[:, 1] >= config.DECISION_THRESHOLD
    assert recall_score(y, predictions) >= config.MIN_RECALL


def test_probabilities_are_valid(model, test_set):
    # toutes les probabilités prédites sont comprises entre 0 et 1
    X, _ = test_set
    proba = model.predict_proba(X)
    assert ((proba >= 0) & (proba <= 1)).all()
