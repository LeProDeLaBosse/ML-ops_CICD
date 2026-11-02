# credit-api — service de scoring de risque de défaut de crédit

Service FastAPI qui expose un modèle scikit-learn de prédiction du risque de
défaut de crédit. Point de départ du **TP noté de la séance 4** : les
consignes sont dans [TP_GUIDE.md](TP_GUIDE.md).

## Membres du groupe

- BYTYQI Dardan
- VIALA Hugo

## Démarrage rapide

```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements-dev.txt
ruff check .                      # lint
pytest                            # tests
uvicorn src.api:app --reload      # API sur http://127.0.0.1:8000/docs
```

## Structure

```
data/        jeux de données (référence, test figé, lot de production)
models/      modèle entraîné (model.joblib) et ses métadonnées
src/         code du service (api.py) et constantes partagées (config.py)
scripts/     entraînement (train.py), vérification d'un service déployé (smoke_test.py)
tests/       tests de l'API (fournis), des données et du modèle (à compléter)
Dockerfile   squelette à compléter
```
