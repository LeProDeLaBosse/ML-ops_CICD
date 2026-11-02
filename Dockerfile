# Squelette de Dockerfile pour le service credit-api — À COMPLÉTER.
# Objectif : une image qui démarre l'API sur le port 8000 avec le modèle
# fourni. Les commentaires TODO indiquent les étapes attendues.

# TODO : image de base Python 3.12 légère, version épinglée
FROM python:3.12-slim

WORKDIR /app

# TODO : installer les dépendances du service (requirements.txt uniquement,
#        pas les outils de dev), en tirant parti du cache des couches

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TODO : copier le code (src/) et le modèle (models/)

COPY src/ src/
COPY models/ models/

# TODO : ne pas exécuter le service en root

RUN useradd --no-create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# TODO : démarrer l'API avec uvicorn (module src.api, objet app),
#        accessible depuis l'extérieur du conteneur, port 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
