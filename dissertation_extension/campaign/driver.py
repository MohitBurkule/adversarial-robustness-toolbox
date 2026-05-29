"""
Campaign driver.

Builds a manifest of >=500 experiments across the three themes, then runs them
one by one. Each experiment writes a human-readable .txt report and appends a
JSON line of metrics. Progress is checkpointed so the campaign is resumable
after any interruption (machine reset, OOM, etc.). Results are committed and
pushed to GitHub periodically so nothing is ever lost.

Usage:
    .venv/bin/python -m campaign.driver            # run everything pending
    .venv/bin/python -m campaign.driver --list     # just print the manifest size
"""
import os
import sys
import json
import time
import argparse
import traceback
import subprocess

from . import common as C
from . import theme_a_unlearning as A
from . import theme_b_learning_rules as B
from . import theme_c_illusions as Cillu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # dissertation_extension/
OUT_DIR = os.path.join(EXT, "results", "campaign")
META_JSONL = os.path.join(OUT_DIR, "metrics.jsonl")
PROGRESS = os.path.join(OUT_DIR, "progress.json")

THEMES = {"A_unlearning": A, "B_learning_rules": B, "C_illusions": Cillu}

COMMIT_EVERY = 5          # commit after this many completed experiments
PUSH_EVERY = 15           # push after this many commits-worth of experiments


def build_manifest():
    specs = []
    for mod in (A, B, Cillu):
        specs.extend(mod.build_specs())
    return specs


def _module_for(spec):
    return THEMES[spec["theme"]]


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_progress(done):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done": sorted(done), "updated": time.time()}, f)
    os.replace(tmp, PROGRESS)


def git(*args, timeout=180):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=timeout)
    except Exception as e:
        print(f"  [git error] {e}", flush=True)
        return None


def commit(msg, push=False):
    git("add", "dissertation_extension/results/campaign")
    r = git("commit", "-q", "-m", msg)
    if r is not None and r.returncode == 0:
        print(f"  [commit] {msg}", flush=True)
    if push:
        pr = git("push", "origin", "dissertation-extension", timeout=300)
        if pr is not None:
            ok = pr.returncode == 0
            print(f"  [push] {'ok' if ok else 'FAILED: ' + (pr.stderr or '')[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run at most N pending (0 = all)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = build_manifest()
    # write/refresh the manifest for transparency
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump([{"theme": s["theme"], "id": s["id"]} for s in specs], f, indent=0)

    if args.list:
        from collections import Counter
        c = Counter(s["theme"] for s in specs)
        print(f"manifest total = {len(specs)}")
        for k, v in c.items():
            print(f"  {k}: {v}")
        return

    done = load_progress()
    pending = [s for s in specs if s["id"] not in done]
    print(f"[campaign] total={len(specs)} done={len(done)} pending={len(pending)}", flush=True)

    n_since_commit = 0
    n_since_push = 0
    completed_now = 0
    t_start = time.time()

    for spec in pending:
        if args.limit and completed_now >= args.limit:
            break
        sid = spec["id"]
        theme = spec["theme"]
        tdir = os.path.join(OUT_DIR, theme)
        os.makedirs(tdir, exist_ok=True)
        out_path = os.path.join(tdir, f"{sid}.txt")
        t0 = time.time()
        try:
            mod = _module_for(spec)
            report, metrics = mod.run(spec)
            dt = time.time() - t0
            with open(out_path, "w") as f:
                f.write(report + f"\n\n[runtime {dt:.1f}s]\n")
            rec = {"id": sid, "theme": theme, "runtime_s": round(dt, 1), **metrics}
            with open(META_JSONL, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"[ok] {sid}  ({dt:.1f}s)  ({completed_now+1} this run)", flush=True)
        except Exception:
            tb = traceback.format_exc()
            with open(out_path, "w") as f:
                f.write(f"FAILED {sid}\n\n{tb}\n")
            with open(META_JSONL, "a") as f:
                f.write(json.dumps({"id": sid, "theme": theme, "error": True}) + "\n")
            print(f"[FAIL] {sid}\n{tb}", flush=True)

        # free GPU between experiments
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        done.add(sid)
        save_progress(done)
        completed_now += 1
        n_since_commit += 1
        n_since_push += 1

        if n_since_commit >= COMMIT_EVERY:
            do_push = n_since_push >= PUSH_EVERY
            commit(f"campaign: +{n_since_commit} results ({len(done)}/{len(specs)} done)", push=do_push)
            n_since_commit = 0
            if do_push:
                n_since_push = 0

    # final flush
    commit(f"campaign: batch complete ({len(done)}/{len(specs)} done)", push=True)
    print(f"[campaign] finished this run: {completed_now} experiments in "
          f"{(time.time()-t_start)/60:.1f} min; total done {len(done)}/{len(specs)}", flush=True)


if __name__ == "__main__":
    main()
