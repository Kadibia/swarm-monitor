"""
monitors.py

Three monitors, all returning a score per agent-round where higher means
"this agent is drifting".

  ProbeMonitor   logistic regression on activations (reads the model internals)
  TfidfMonitor   logistic regression on message text (a generic log-only monitor)
  oracle_scores  surface signature of the known drifts in the text (a log-only
                 upper bound for in-distribution drifts, it knows both signatures)
"""

import glob
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(run_dir: str):
    rows, acts = [], []
    files = sorted(glob.glob(os.path.join(run_dir, "ep_*.json")))
    if not files:
        raise FileNotFoundError(f"No episodes found in {run_dir}. Run generate.py first.")
    for jf in files:
        with open(jf, encoding="utf-8") as f:
            r = json.load(f)
        a = np.load(jf[:-5] + ".npy")
        assert len(r) == len(a), f"row/activation mismatch in {jf}"
        rows += r
        acts.append(a)
    return rows, np.concatenate(acts)


class ProbeMonitor:
    def __init__(self, C: float = 0.1):
        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=3000)
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class TfidfMonitor:
    def __init__(self, C: float = 1.0):
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(C=C, max_iter=3000),
        )

    def fit(self, texts, y):
        self.model.fit(list(texts), y)
        return self

    def score(self, texts):
        return self.model.predict_proba(list(texts))[:, 1]


def oracle_scores(rows):
    return np.array([float(r["signature"]) for r in rows])
