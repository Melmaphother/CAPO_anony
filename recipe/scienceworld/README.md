# ScienceWorld Recipe

This recipe gives the policy official state-conditioned action templates and visible object referents, with a 30-agent-action horizon. The simulator itself uses a 100-step limit, so the 30-step agent-flow cutoff is not treated as an environment terminal state. The full valid-command list is used only to label whether a proposed command matches its canonical form; the parser also accepts valid surface forms such as `go kitchen`. The list is not exposed in the prompt and does not prevent a command from reaching the environment. The prompt keeps an all-action ledger plus the most recent two observation/action pairs.

## Environment and data preparation

Install ScienceWorld and Java in an isolated environment, then point the recipe to that checkout:

```bash
python -m pip install -e /path/to/ScienceWorld
export SCIENCEWORLD_HOME=/path/to/ScienceWorld
python recipe/scienceworld/prepare_scienceworld_arft.py --output-dir data/scienceworld
```

The manifests contain 3,592 training examples, 1,819 test examples, and a deterministic task-stratified 100-sample `test_in_train` subset. Training monitors only `test_in_train`; the full test split is reserved for separate checkpoint evaluation. Task descriptions are resolved from the simulator at rollout time, so manifest preparation does not start one Java process per row.

## Launch

```bash
export SCIENCEWORLD_HOME=/path/to/ScienceWorld
bash examples/scienceworld/run_ppo.sh
```

Training uses a terminal reward: a terminal episode with official `final_score > 0` receives `+10.0` on its final action, while every other action receives `0.0`. Official score changes are retained as diagnostics, and evaluation reports terminal positive-score success separately from the official final score.
