"""Vérifie qu'un service credit-api déployé répond correctement.

Usage :
    python scripts/smoke_test.py --url http://localhost:8000

Code de sortie 0 si le service est sain, 1 sinon : utilisable tel quel
comme étape de pipeline. N'utilise que la bibliothèque standard (aucune
installation nécessaire).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SAMPLE_REQUEST = {
    "age": 42,
    "income": 38000,
    "loan_amount": 12000,
    "employment_years": 6.5,
    "num_credit_lines": 3,
    "region": "North",
}


def call(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def wait_until_up(base_url, retries, delay):
    for attempt in range(1, retries + 1):
        try:
            status, body = call(f"{base_url}/health")
            if status == 200:
                return body
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        print(f"[{attempt}/{retries}] service pas encore disponible, nouvel essai dans {delay}s")
        time.sleep(delay)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    health = wait_until_up(base_url, args.retries, args.delay)
    if health is None:
        print(f"ÉCHEC : {base_url}/health ne répond pas")
        sys.exit(1)
    print(f"/health OK : {health}")

    status, body = call(f"{base_url}/predict", SAMPLE_REQUEST)
    proba = body.get("default_probability")
    if status != 200 or not isinstance(proba, (int, float)) or not 0 <= proba <= 1:
        print(f"ÉCHEC : réponse inattendue de /predict : {status} {body}")
        sys.exit(1)
    print(f"/predict OK : {body}")


if __name__ == "__main__":
    main()
