import argparse
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

from src import config

PRODUCTION_PATH = config.ROOT / "data" / "production_batch.csv"
PSI_ALERT = 0.25
PSI_WATCH = 0.1


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference), np.asarray(current)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference, current):
    cats = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    cur = current.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def predict(base_url, row):
    payload = json.dumps(row).encode()
    req = urllib.request.Request(
        f"{base_url}/predict", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())["default_probability"]


def status(value):
    return "alert" if value > PSI_ALERT else "watch" if value > PSI_WATCH else "stable"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="drift_report.json")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    reference = pd.read_csv(config.REFERENCE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    production_scores = [predict(base_url, row) for row in production.to_dict("records")]
    reference_scores = [predict(base_url, row) for row in reference.head(500).to_dict("records")]

    values = {col: psi(reference[col], production[col]) for col in config.NUMERIC_FEATURES}
    values["region"] = psi_categorical(reference["region"], production["region"])
    values["predicted_score"] = psi(reference_scores, production_scores)

    report = {name: {"psi": round(v, 4), "status": status(v)} for name, v in values.items()}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    drifted = [name for name, r in report.items() if r["status"] == "alert"]
    if drifted:
        print(f"DÉRIVE significative : {', '.join(drifted)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
