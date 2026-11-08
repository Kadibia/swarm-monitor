"""decoy_check.py
Five-fold cross-validated probe scores, grouped by episode, for four row groups.
Usage  python decoy_check.py --run data/run1
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from monitors import ProbeMonitor, load_data

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="data/run1")
args = ap.parse_args()

rows, X = load_data(args.run)
y = np.array([int(r["drift_active"]) for r in rows])
groups = np.array([r["episode_id"] for r in rows])
drift = y == 1
decoy = np.array([r["decoy_active"] for r in rows])
visible = np.array([r["behavior_exhibited"] for r in rows])
hidden = drift & ~visible
none = ~drift & ~decoy

scores = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    scores[te] = ProbeMonitor().fit(X[tr], y[tr]).score(X[te])

for name, m in [("drift, behavior visible", drift & visible),
                ("drift, behavior hidden", hidden),
                ("decoy instruction", decoy),
                ("no instruction", none)]:
    print(f"{name:24s} rows {m.sum():4d}  mean score {scores[m].mean():.3f}")
m = drift | decoy
print("AUROC, drift rows vs decoy rows:", round(roc_auc_score(y[m], scores[m]), 3))
