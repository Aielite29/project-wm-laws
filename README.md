# Rule Discovery in Gridworlds

This repo contains two main scripts:

- **`dyna_law_env2.py`**: Key-Lock Gridworld environment + Dyna-Q agent. Supports multiple dynamic "laws", logs episodes/trajectories, saves results, and generates plots & imagination GIFs.  
- **`vlm_rule_discovery.py`**: Vision-Language Model (VLM) pipeline for discovering symbolic rules from episodes. Includes verification, clustering, refinement, and visualizations.

---

## Features

### `dyna_law_env2.py`
- Key-Lock environment with different rule sets:
  - `SimpleAdventureLaw`, `LockRequiresSwordLaw`, `AggressiveMonsterLaw`, `ProbabilisticMonsterLaw`, `FleeingMonsterLaw`
- Dyna-Q agent with (optional) prioritized replay.
- Saves:
  - `episodes/episode_*.txt`
  - `trajectories_runX.jsonl`
  - `rule_archive_runX.jsonl`
  - `q_table_runX.npz` (+ `.json`)
  - `train_rewards_runX.npy`
  - `results_runX.json`, `summary_runX.json`
  - `plots/*.png`, `imagination/*.gif`

### `vlm_rule_discovery.py`
- Parses episodes → symbolic facts.
- Queries VLM (`dry`, `http`, or `openai`).
- Extracts, verifies, clusters, and refines rules.
- Saves:
  - `rules_vlm.jsonl`, `canonical.jsonl`, `rules_summary.json`
  - `vlm_viz/*.png` (plots, heatmaps)
  - `vlm_viz/rule_examples/` (snapshots)

---

Usage
1. Train & Log Episodes
python dyna_law_env2.py --episodes 500 --width 8 --height 8 \
    --out_dir ./runs/exp1 --mode fixed --fixed_law 0

2. Discover Rules

Dry-run:

python vlm_rule_discovery.py --run_dir ./runs/exp1 --vlm dry \
    --out rules_vlm.jsonl --out_summary rules_summary.json


HTTP VLM:

python vlm_rule_discovery.py --run_dir ./runs/exp1 --vlm http \
    --vlm_url http://localhost:8000/predict


OpenAI:

python vlm_rule_discovery.py --run_dir ./runs/exp1 --vlm openai \
    --model gpt-4o-mini

Workflow

Train agent with dyna_law_env2.py → generate episodes/trajectories.

Run vlm_rule_discovery.py → extract/verify rules.

Inspect outputs in rules_*.json + vlm_viz/.
