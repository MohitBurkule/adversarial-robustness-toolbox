# Project Instructions

## Agent Workflow

- **All tasks must be executed via background agents** — never run scripts directly in the main session
- Use `Agent(run_in_background=True)` for all script execution, fixing, and research tasks
- Main session only manages agents: dispatches tasks, monitors completions, reports results
- **For harder tasks (multi-file fixes, research, complex debugging): use `model="opus"`**

## Project Context

- Dataset: Fashion-MNIST (all experiments)
- Environment: `.venv/bin/python` in `/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension/`
- All hypothesis scripts go in `hypotheses/`, results in `results/fashion_mnist/`
- Campaign common utilities in `campaign/common.py`
- Results summary at `RESULTS_SUMMARY.md`
