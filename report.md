# [Title]

## Claim
[One sentence. Example, "On a small three-agent swarm, a linear probe on activations detects planted drift in cases where log-only monitors fail, and keeps working when the logs are tampered with." Edit this to match what you actually found.]

## Setup
- Model and layer [fill in]
- Sandbox [three agents, shared board, 8 rounds, summary task, passages]
- Drift types [marker, topic] and decoys [list]
- Episodes [n total, n test drift episodes]
- Monitors [probe, log tfidf, log oracle]
- Alarm threshold [5 percent false alarm budget set on validation]

[Insert a small diagram of the sandbox here.]

## Results
**Figure 1.** [fig1_detection.png] [One or two sentences on what it shows.]

**Figure 2.** [fig2_tamper.png] [Deletion rate used and what happened.]

**Unseen drift.** [Probe AUROC and log tfidf AUROC in both directions, from results.json.]

**Hidden behavior.** [hidden_tpr for each monitor and the number of rows, from results.json.]

## Limitations
- Planted drift from an explicit instruction, not drift that emerges naturally.
- Small open-weight model, results may not transfer to frontier agents.
- Small test set, [n] drift episodes, so small gaps are noise.
- The probe may partly read the presence of an instruction, only partly controlled by the decoy.
- [Add anything you saw in the raw transcripts.]

## What a follow-up would test
- More drift types and covert drifts that do not leave a surface signature.
- Agents told to conceal their behavior.
- Real logs from deployed agent systems.
- Probes trained on one model and tested on another.

## Reproduce
[Paste the exact commands from the README.]
