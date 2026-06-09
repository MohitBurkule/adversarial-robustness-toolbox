# Project Instructions

## Agent Workflow

- **The main session does MANAGEMENT ONLY** — never write scripts, run scripts, fix code, or do research directly. Every unit of work is delegated to a background agent.
- **The main session must NEVER run Bash** — not even for quick checks (ls, grep, reading result files, inspecting status). ALL inspection/checking/verification is delegated to a background agent too.
- **All tasks must be executed via background agents** — `Agent(run_in_background=True)` for script writing, execution, fixing, research, summary updates, AND status/result checking
- Main session responsibilities: dispatch agents, receive their completion reports, relay results to the user. That's it. Reading a file with the Read tool to brief an agent is fine; running shell commands is not.
- **Model choice:** use `model="sonnet"` for straightforward single-script write+run tasks; use `model="opus"` for harder tasks (multi-file fixes, literature research, complex debugging, novel hypothesis design)

## Project Context

- Dataset: Fashion-MNIST (all experiments)
- Environment: `.venv/bin/python` in `/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension/`
- All hypothesis scripts go in `hypotheses/`, results in `results/fashion_mnist/`
- Campaign common utilities in `campaign/common.py`
- Results summary at `RESULTS_SUMMARY.md`
