"""Tests de validation des données — À COMPLÉTER.

Ces tests vérifient que les jeux de données du dépôt (data/reference.csv et
data/test_set.csv) respectent le contrat de données défini dans
src/config.py : FEATURES, TARGET, ALLOWED_REGIONS, VALUE_RANGES.

Ils doivent échouer si quelqu'un commite des données corrompues.
"""
import pandas as pd
import pytest

from src import config

DATASETS = [config.REFERENCE_PATH, config.TEST_SET_PATH]


@pytest.fixture(params=DATASETS, ids=lambda p: p.name)
def df(request):
    return pd.read_csv(request.param)


def test_expected_columns(df):
    # le jeu contient exactement les colonnes FEATURES + [TARGET]
    assert set(df.columns) == set(config.FEATURES + [config.TARGET])



def test_no_missing_values(df):
    # aucune valeur manquante
    assert not df.isna().any().any()


def test_values_in_range(df):
    # chaque colonne numérique respecte VALUE_RANGES (bornes incluses)
    for column, (low, high) in config.VALUE_RANGES.items():
        assert df[column].between(low, high).all(), f"{column} hors bornes"


def test_regions_are_allowed(df):
    # la colonne region ne contient que des valeurs de ALLOWED_REGIONS
    assert df["region"].isin(config.ALLOWED_REGIONS).all()


def test_target_is_binary(df):
    # la cible ne contient que 0 et 1
    assert df[config.TARGET].isin([0, 1]).all()