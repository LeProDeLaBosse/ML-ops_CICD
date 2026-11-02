"""Génère les jeux de données du TP noté : scoring de risque de défaut de crédit.

Usage :
    python data/generate_dataset.py

Produit trois fichiers (déjà fournis dans le dépôt, ce script sert seulement
à les régénérer) :
    data/reference.csv         échantillon des données d'entraînement (référence)
    data/test_set.csv          jeu de test figé, utilisé pour valider le modèle
    data/production_batch.csv  requêtes « récentes » reçues en production (monitoring)
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 5000
REGIONS = np.array(["North", "South", "East", "West"])


def generate(n, seed, income_shift=0.0, region_probs=(0.35, 0.15, 0.25, 0.25)):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 76, size=n)
    income = (rng.normal(38000, 14000, size=n) + income_shift).clip(8000, None)
    loan_amount = rng.normal(12000, 6000, size=n).clip(500, None)
    employment_years = rng.normal(7, 5, size=n).clip(0, None)
    num_credit_lines = rng.integers(0, 9, size=n)
    region = rng.choice(REGIONS, size=n, p=list(region_probs))

    z = (
        -2.6
        + 0.09 * (loan_amount / 1000)
        - 0.045 * (income / 1000)
        - 0.08 * employment_years
        + 0.22 * num_credit_lines
        + np.where(region == "South", 0.35, 0.0)
        + rng.normal(0, 0.5, size=n)
    )
    default = rng.binomial(1, 1 / (1 + np.exp(-z)))

    return pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "employment_years": employment_years.round(1),
            "num_credit_lines": num_credit_lines,
            "region": region,
            "default": default,
        }
    )


if __name__ == "__main__":
    data = generate(N_SAMPLES, RANDOM_SEED)
    reference = data.sample(frac=0.8, random_state=RANDOM_SEED)
    test_set = data.drop(reference.index)
    reference.to_csv("data/reference.csv", index=False)
    test_set.to_csv("data/test_set.csv", index=False)

    # Requêtes de production : revenus en baisse et répartition régionale
    # différente -> dérive volontaire, à détecter par le monitoring.
    production = generate(
        1000, RANDOM_SEED + 1, income_shift=-9000, region_probs=(0.2, 0.4, 0.2, 0.2)
    ).drop(columns="default")
    production.to_csv("data/production_batch.csv", index=False)

    for name, df in [("reference.csv", reference), ("test_set.csv", test_set)]:
        print(f"{name} : {len(df)} lignes, taux de défaut {df['default'].mean():.3f}")
    print(f"production_batch.csv : {len(production)} lignes")
