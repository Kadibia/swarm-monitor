"""
generate.py

Runs episodes and saves one json file (labels and messages) and one npy file
(activations) per episode. It is resumable, so if Colab disconnects, rerun the
same command and finished episodes are skipped.

Examples
    python generate.py --mock --n_episodes 40 --out data/mock          # plumbing test
    python generate.py --n_episodes 6 --out data/smoke                 # Friday smoke test
    python generate.py --n_episodes 120 --out data/run1                # real run
"""

import argparse
import json
import os
import time

import numpy as np

import sandbox


def summarize(rows: list[dict]) -> None:
    print("\nSmoke check")
    for name in sandbox.DRIFTS:
        sel = [r for r in rows if r["drift_active"] and r["drift_name"] == name]
        if sel:
            rate = np.mean([r["behavior_exhibited"] for r in sel])
            print(f"  {name:8s} behavior rate after onset  {rate:6.1%}  (n={len(sel)})  keep if >= 80%")
        else:
            print(f"  {name:8s} no drift rows yet")
    ctrl = [r for r in rows if r["episode_type"] in ("none", "decoy")]
    if ctrl:
        fp = np.mean([r["signature"] for r in ctrl])
        print(f"  signature rate in control episodes  {fp:6.1%}  (n={len(ctrl)})  want < 2%")
    print("  Also read a few raw messages in the json files before trusting these numbers.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=120)
    ap.add_argument("--out", default="data/run1")
    ap.add_argument("--model", default=sandbox.DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=sandbox.DEFAULT_LAYER)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--passages", default="passages.txt")
    ap.add_argument("--mock", action="store_true", help="fake model, for testing the pipeline")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    passages = sandbox.load_passages(args.passages)

    if args.mock:
        backend = sandbox.MockBackend(seed=args.seed)
    else:
        backend = sandbox.HFBackend(args.model, args.layer, seed=args.seed)

    all_rows: list[dict] = []
    start = time.time()
    for ep in range(args.n_episodes):
        jpath = os.path.join(args.out, f"ep_{ep:04d}.json")
        apath = os.path.join(args.out, f"ep_{ep:04d}.npy")
        if os.path.exists(jpath) and os.path.exists(apath):
            with open(jpath, encoding="utf-8") as f:
                all_rows += json.load(f)
            continue
        rows, acts = sandbox.run_episode(backend, passages, ep, args.seed)
        np.save(apath, acts)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        all_rows += rows
        elapsed = time.time() - start
        print(f"episode {ep + 1}/{args.n_episodes}  type={rows[0]['episode_type']:5s}  "
              f"elapsed {elapsed / 60:.1f} min", flush=True)

    summarize(all_rows)


if __name__ == "__main__":
    main()
