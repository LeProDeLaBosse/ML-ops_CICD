# TP noté (groupe) — Pipeline CI/CD complète pour un service de prédiction

**Séance 4 — MLOps — M2** · Groupes de 3 à 4 · **70 % de la note finale**

## Objectif

En groupe et en temps limité, concevoir et faire fonctionner sur **GitLab**
une pipeline CI/CD complète pour `credit-api`, le service de prédiction du
risque de défaut de crédit (le cas de l'atelier de la séance 1, cette fois
avec un code propre et un modèle déjà entraîné).

À la fin de la séance, chaque push ou merge request sur votre dépôt doit
déclencher automatiquement une pipeline qui :

1. **vérifie** le code, les données et le modèle (lint, tests unitaires,
   validation des données, validation du modèle) ;
2. **construit** l'image Docker du service et la **pousse** dans le
   Container Registry de votre projet GitLab ;
3. **déploie** (en simulation) le service en *staging* et vérifie qu'il
   répond ;
4. permet une **mise en production** déclenchée manuellement depuis la
   branche principale ;
5. exécute un **monitoring** basique après déploiement.

Et surtout, elle doit **bloquer** toute modification qui casse quelque chose.

## Outils utilisés — gratuits et open-source

| Outil | Rôle |
|---|---|
| GitLab (dépôt, GitLab CI/CD, Container Registry, Environments) | Hébergement du code, exécution de la pipeline, stockage des images |
| Docker | Conteneurisation du service (en local et dans la pipeline) |
| Python 3.12, FastAPI, scikit-learn, pandas | Service de prédiction et modèle |
| pytest, ruff | Tests et lint |

## Ce qui est fourni

```
data/reference.csv          échantillon des données d'entraînement (4 000 lignes)
data/test_set.csv           jeu de test figé, avec labels (1 000 lignes)
data/production_batch.csv   requêtes reçues en production (monitoring)
models/model.joblib         modèle entraîné (pipeline scikit-learn) — NE PAS réentraîner
models/metadata.json        version du modèle, métriques, version de scikit-learn
src/config.py               contrat de données et seuils de validation du modèle
src/api.py                  API FastAPI complète : /health et /predict
scripts/smoke_test.py       vérifie qu'un service déployé répond (code de sortie 0/1)
scripts/train.py            script d'entraînement (pour information)
tests/test_api.py           tests de l'API — fournis, ils passent
tests/test_data.py          tests de validation des données — À COMPLÉTER
tests/test_model.py         tests de validation du modèle — À COMPLÉTER
Dockerfile                  squelette — À COMPLÉTER
requirements.txt            dépendances du service (versions figées)
requirements-dev.txt        dépendances de développement et de CI
```

> **Ne modifiez pas les versions de `requirements.txt`.** Le modèle a été
> sérialisé avec scikit-learn 1.9.1 : le charger avec une autre version
> n'est pas garanti.

Il n'y a **pas** de `.gitlab-ci.yml` : c'est à vous de l'écrire.

## Mise en route

```bash
cd tp_note_pipeline_cicd
python -m venv venv
source venv/bin/activate            # Windows : venv\Scripts\activate
pip install -r requirements-dev.txt
ruff check .                        # doit afficher "All checks passed!"
pytest                              # 3 tests passent, les autres sont "skipped"
uvicorn src.api:app --reload        # puis ouvrir http://127.0.0.1:8000/docs
```

---

## Étape 0 — Créer le dépôt GitLab du groupe

1. **Une seule personne** crée un projet GitLab **privé** et vide (sans
   README), par exemple `mlops-tp`.
2. Ajoutez les autres membres (*Manage → Members*) avec le rôle
   **Maintainer**, et l'enseignante avec le rôle **Developer** (pour
   accéder au dépôt, aux pipelines et au registry lors de la correction).
3. Poussez le contenu de ce dossier :
   ```bash
   cd tp_note_pipeline_cicd
   git init -b main
   git add .
   git commit -m "Code de départ du TP"
   git remote add origin <URL de votre projet>
   git push -u origin main
   ```
   - Copiez l'URL depuis le bouton **Code** de la page du projet (*Clone
     with HTTPS* ou *Clone with SSH*), pas depuis la barre d'adresse du
     navigateur.
4. Dans les paramètres du projet :
   - *Settings → General → Visibility, project features, permissions* :
     vérifiez que **CI/CD** et **Container registry** sont activés ;
   - *Settings → CI/CD → Runners* : vérifiez qu'au moins un runner est
     disponible pour votre projet ;
   - *Settings → Merge requests → Merge checks* : cochez **Pipelines must
     succeed** — une merge request dont la pipeline échoue ne doit pas
     pouvoir être fusionnée.
5. Complétez la liste des membres dans `README.md`.

Organisez-vous : le travail se parallélise bien (tests / Dockerfile /
pipeline), mais **une seule branche principale** fait foi. Travaillez sur
des branches et fusionnez par merge request — c'est aussi le meilleur moyen
de tester votre pipeline.

---

## Étape 1 — Compléter les tests

Les tests sont le cœur de la pipeline : une étape qui ne teste rien ne
bloque rien.

**`tests/test_data.py`** — les deux jeux `data/reference.csv` et
`data/test_set.csv` doivent respecter le contrat défini dans
`src/config.py` :

- exactement les colonnes `FEATURES` + `TARGET` ;
- aucune valeur manquante ;
- chaque colonne numérique dans les bornes de `VALUE_RANGES` (incluses) ;
- `region` uniquement parmi `ALLOWED_REGIONS` ;
- cible binaire (0 ou 1).

**`tests/test_model.py`** — le modèle `models/model.joblib`, évalué sur
`data/test_set.csv`, doit avoir :

- une ROC AUC ≥ `MIN_ROC_AUC` ;
- un rappel de la classe 1 (seuil `DECISION_THRESHOLD`) ≥ `MIN_RECALL` ;
- des probabilités prédites toutes comprises entre 0 et 1.

Les squelettes, fixtures et `TODO` sont en place : remplacez chaque
`pytest.skip(...)` par de vraies assertions (`sklearn.metrics` fournit
`roc_auc_score` et `recall_score`).

**Vérification :** `pytest` doit afficher tous les tests en *passed*, aucun
en *skipped*. Faites ensuite l'expérience : modifiez temporairement une
valeur de `data/test_set.csv` (par exemple une région `Nord`) et vérifiez
qu'un test échoue. Annulez la modification.

---

## Étape 2 — Conteneuriser le service

Complétez le `Dockerfile` en suivant les `TODO`. Bonnes pratiques
attendues : image de base légère et épinglée, ordre des couches
pensé pour le cache, uniquement les dépendances du service (pas celles de
dev), utilisateur non-root, démarrage par `uvicorn` en forme exec. Un
`.dockerignore` adapté est un plus.

**Vérification en local :**

```bash
docker build -t credit-api:dev .
docker run --rm -p 8000:8000 credit-api:dev
# dans un autre terminal :
python scripts/smoke_test.py --url http://localhost:8000
```

Le smoke test doit afficher `/health OK` puis `/predict OK`.

---

## Étape 3 — Écrire la pipeline `.gitlab-ci.yml`

Créez `.gitlab-ci.yml` à la racine du dépôt. Pas de solution toute faite
ici : appuyez-vous sur le cours et sur la
documentation officielle.

**Documentation :**
[docs.gitlab.com/ci/yaml](https://docs.gitlab.com/ci/yaml/) (référence des mots-clés) ·
[docs.gitlab.com/ci/variables/predefined_variables](https://docs.gitlab.com/ci/variables/predefined_variables/) (variables prédéfinies) ·
[docs.gitlab.com/ci/services](https://docs.gitlab.com/ci/services/) (services)

### Exigences minimales

| Étape | Attendu |
|---|---|
| **Déclenchement** | Automatique sur chaque push et sur chaque merge request — sans pipelines en double pour un même commit |
| **Lint** | `ruff check .` |
| **Tests unitaires** | `tests/test_api.py` |
| **Validation des données** | `tests/test_data.py`, dans un job distinct |
| **Validation du modèle** | `tests/test_model.py`, dans un job distinct |
| **Build & push** | Image construite depuis votre `Dockerfile`, poussée dans le Container Registry du projet, taguée avec le SHA du commit |
| **Déploiement staging** | L'image qui vient d'être construite est démarrée et `scripts/smoke_test.py` vérifie qu'elle répond ; job rattaché à un environnement GitLab `staging` |
| **Déploiement production** | Sur la branche principale uniquement, **déclenché manuellement** ; il « promeut » l'image testée en la retaguant `production` dans le registry ; job rattaché à un environnement `production` |
| **Monitoring post-déploiement** | Après le staging : l'image déployée reçoit `data/production_batch.csv`, la dérive (features et scores prédits) est comparée à la référence, un rapport est publié en artefact, et la dérive est signalée **sans bloquer** la pipeline |
| **Enchaînement** | Une étape en échec bloque toutes les suivantes : aucune image n'est construite si un test échoue, rien n'est déployé si le build ou le staging échoue |

### Repères GitLab utiles

- **Variables prédéfinies** pour le registry, à utiliser plutôt que des
  valeurs en dur : `$CI_REGISTRY`, `$CI_REGISTRY_IMAGE`, `$CI_REGISTRY_USER`,
  `$CI_REGISTRY_PASSWORD`, et `$CI_COMMIT_SHORT_SHA` pour le tag.
  **Aucun identifiant ne doit apparaître en clair dans le dépôt.**
- **Construire une image dans un job** : les runners du GitLab de l'école
  ne sont **pas** en mode privilégié, donc *Docker-in-Docker*
  (`docker:dind`) n'y fonctionne pas. Utilisez **Kaniko**, qui construit et
  pousse une image à partir de votre `Dockerfile` sans daemon Docker. Le
  fragment ci-dessous est à utiliser dans votre `.gitlab-ci.yml` ; il
  vous reste à compléter les `TODO` :
  ```yaml
  # Identifiants du registry, réutilisables par Kaniko et crane
  .registry-auth: &registry-auth
    - mkdir -p "$DOCKER_CONFIG"
    - AUTH=$(printf "%s:%s" "$CI_REGISTRY_USER" "$CI_REGISTRY_PASSWORD" | base64 | tr -d '\n')
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"auth\":\"$AUTH\"}}}" > "$DOCKER_CONFIG/config.json"

  build-image:
    # TODO : stage
    image:
      name: gcr.io/kaniko-project/executor:v1.24.0-debug
      entrypoint: [""]
    variables:
      DOCKER_CONFIG: /kaniko/.docker
    before_script: *registry-auth
    script:
      - /kaniko/executor
        --context "$CI_PROJECT_DIR"
        --dockerfile "$CI_PROJECT_DIR/Dockerfile"
        --destination "TODO : image et tag, à réutiliser à l'identique dans services:"
  ```
  Points d'attention :
  - `entrypoint: [""]` n'est valide qu'avec la forme développée
    `image: name: …` montrée ici ;
  - le nom et le tag passés à `--destination` doivent être **exactement**
    ceux que vous utiliserez dans `services:` pour le staging et le
    monitoring (même variable de SHA, courte ou longue) ;
  - Kaniko ne peut pas être testé en local : vérifiez d'abord votre
    `Dockerfile` avec `docker build` (étape 2).
- **Retaguer une image sans daemon** (promotion en production) : l'outil
  **crane** (image `gcr.io/go-containerregistry/crane:debug`, même forme
  `image: name: … entrypoint: [""]`) fait `crane tag <image:tag> production`
  directement dans le registry. Réutilisez `before_script: *registry-auth`
  avec la variable `DOCKER_CONFIG: /tmp/.docker` sur le job.
- **Démarrer votre image pendant un job** pour le staging : le mot-clé
  `services:` accepte une image de votre registry, avec un `alias:` qui
  devient le nom d'hôte joignable depuis le job
  (`http://<alias>:8000`).
- **Ordonner** : `stages:` (ordre global) et, si besoin, `needs:`.
- **Conditionner** : `rules:` (avec `if:` et `when: manual`) ;
  `workflow: rules:` pour décider quand une pipeline est créée.
- **Environnements** : `environment: name: staging` — ils apparaissent
  ensuite dans *Operate → Environments*.
- **Factoriser** : un job caché (nom commençant par `.`) + `extends:`
  évite de répéter l'image et l'installation des dépendances dans chaque job.
- **Accélérer** : `cache:` sur le dossier de cache de pip.
- **Rapports de tests** : `pytest --junitxml=report.xml` +
  `artifacts: reports: junit:` affiche les tests dans la merge request.

**Vérification :** l'éditeur de pipeline de GitLab (*Build → Pipeline
editor*) valide la syntaxe avant même de commiter. Puis, sur `main`,
toutes les étapes doivent être vertes, et le bouton ▶ du job de production
doit permettre de déployer. Vérifiez dans *Deploy → Container Registry* que
les images (tag SHA et tag `production`) sont bien présentes.

---

## Monitoring post-déploiement

Ajoutez, après le staging, un job qui :

- démarre l'image déployée (comme pour le staging) ;
- envoie au service les requêtes de `data/production_batch.csv` ;
- compare au jeu de référence la distribution des features et des scores
  prédits (par exemple avec le PSI vu en séance 1 : < 0,1 stable,
  0,1–0,25 à surveiller, > 0,25 dérive significative) ;
- publie un rapport (JSON ou HTML) en **artefact** du job ;
- signale la dérive **sans bloquer** la pipeline (`allow_failure: true`).

Indice : le lot de production a été construit avec une dérive volontaire.
Votre monitoring doit la détecter.

---

## Évaluation (70 % de la note finale)

| Critère | Pondération | Ce qui est regardé |
|---|---|---|
| Pipeline fonctionnelle | 45 % | Déclenchement automatique (push et MR) ; toutes les étapes exécutées avec succès sur `main` ; un échec bloque bien la suite |
| Complétude des étapes | 30 % | Lint, tests unitaires, validation des données, validation du modèle, build & push, staging avec smoke test, production manuelle, monitoring post-déploiement |
| Qualité et lisibilité du code / de la configuration | 25 % | `.gitlab-ci.yml` structuré et sans duplication, Dockerfile conforme aux bonnes pratiques, aucun secret en clair, tests lisibles |

**Rendu :** l'état de la branche principale de votre dépôt GitLab à la fin de la séance. Rien d'autre à envoyer.

---

## Dépannage

- **`Cannot connect to the Docker daemon` dans le job de build** : vous
  utilisez `docker build` / `docker:dind`, qui ne fonctionne pas sur les
  runners de l'école. Passez à Kaniko (voir *Repères GitLab utiles*).
- **`UNAUTHORIZED` / `error checking push permissions` avec Kaniko** : le
  fichier `config.json` n'existe pas, est mal formé ou n'est pas à
  l'emplacement lu par Kaniko, ou le Container Registry n'est pas activé
  pour le projet.
- **`could not lock config file …/.gitconfig` avant même le script** : vous
  avez redéfini `HOME` dans les `variables:` du job ; GitLab l'applique
  aussi au clonage du dépôt. Ne touchez pas à `HOME`.
- **`git push` redirige vers `users/sign_in`** : l'URL du dépôt distant est
  incorrecte (copiée depuis le navigateur, avec un `#` final par exemple ;
  corrigez-la avec `git remote set-url origin <URL du bouton Code>`), ou
  vous utilisez votre mot de passe au lieu d'un token d'accès personnel.
- **`address already in use` au `docker run -p 8000:8000`** : le port 8000
  de votre machine est déjà pris. Publiez un autre port :
  `docker run --rm -p 8001:8000 credit-api:dev`, puis
  `python scripts/smoke_test.py --url http://localhost:8001`.
- **Le job de staging ne trouve pas le service** : vérifiez que le nom de
  l'image dans `services:` correspond exactement au tag poussé, et que vous
  appelez l'`alias` déclaré, sur le port 8000.
- **Deux pipelines par commit** (branche + MR) : utilisez `workflow: rules:`
  (voir la documentation de `workflow`).
- **`InconsistentVersionWarning` ou erreur au chargement du modèle** : vous
  avez changé la version de scikit-learn. Revenez à `requirements.txt`
  d'origine.
- **Les imports `from src import config` échouent** : lancez les commandes
  depuis la racine du dépôt.
