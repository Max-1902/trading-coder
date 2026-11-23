"""Generate completions for benchmark tasks with MLX. One model per process.

Usage:
  python -m gen.sample --model qwen3b --shots 0 --run baseline_qwen3b_0shot
  python -m gen.sample --model qwen3b --ids ma_cross-00,holding-03 --run slice_test
  python -m gen.sample --model qwen3b --n 5 --temp 0.8 --run qwen3b_pass5

Writes results/<run>/config.json and results/<run>/generations.jsonl.
Refuses to overwrite an existing run folder.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_tasks(split: str) -> list[dict]:
    files = {"test_id": ["tasks_test_id.jsonl"], "test_ood": ["tasks_test_ood.jsonl"],
             "all": ["tasks_test_id.jsonl", "tasks_test_ood.jsonl"]}[split]
    return [json.loads(l) for f in files for l in (ROOT / "bench" / f).read_text().splitlines() if l.strip()]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in gen.models.MODELS or a path / HF id")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="all", choices=["test_id", "test_ood", "all"])
    ap.add_argument("--tasks-file", default=None, help="JSONL of tasks to sample instead of the benchmark")
    ap.add_argument("--ids", default=None, help="comma-separated task ids (slice)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="samples per task")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2, help="small batches keep memory low (16 GB Mac)")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to rest the GPU between batches")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    out = ROOT / "results" / a.run
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten. Pick a new --run name.")

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    from gen.models import MODELS
    from gen.prompts import fewshot_examples, messages

    if a.tasks_file:
        tasks = [json.loads(l) for l in Path(a.tasks_file).read_text().splitlines() if l.strip()]
    else:
        tasks = load_tasks(a.split)
    if a.ids:
        want = a.ids.split(",")
        tasks = [t for t in tasks if t["id"] in want]
    if a.limit:
        tasks = tasks[: a.limit]
    model_path = MODELS.get(a.model, a.model)

    out.mkdir(parents=True)
    config = {k: v for k, v in vars(a).items()}
    config.update({"model_path": model_path, "n_tasks": len(tasks), "task_ids": [t["id"] for t in tasks],
                   "fewshot_ids": [e["id"] for e in fewshot_examples()[: a.shots]],
                   "git_commit": git_commit(), "mlx_lm": mlx_lm.__version__, "mlx": mx.__version__,
                   "machine": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    mx.random.seed(a.seed)
    model, tok = load(model_path, adapter_path=a.adapter)
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)

    jobs = [(t, k) for t in tasks for k in range(a.n)]
    prompts = {t["id"]: tok.apply_chat_template(messages(t["prompt"], a.shots), add_generation_prompt=True)
               for t in tasks}
    t0, done = time.time(), 0
    with open(out / "generations.jsonl", "w") as f:
        for i in range(0, len(jobs), a.batch_size):
            batch = jobs[i : i + a.batch_size]
            resp = batch_generate(model, tok, [prompts[t["id"]] for t, _ in batch], max_tokens=a.max_tokens,
                                  sampler=sampler, return_token_ids=True)
            for (t, k), text, ids in zip(batch, resp.texts, resp.token_ids):
                f.write(json.dumps({"task_id": t["id"], "sample": k, "text": text, "n_tokens": len(ids),
                                    "prompt_tokens": len(prompts[t["id"]]),
                                    "hit_max_tokens": len(ids) >= a.max_tokens}) + "\n")
            f.flush()
            done += len(batch)
            el = time.time() - t0
            print(f"{done}/{len(jobs)}  {el:.0f}s elapsed, ~{el / done * (len(jobs) - done):.0f}s left", flush=True)
            if a.pause and done < len(jobs):
                time.sleep(a.pause)

    config.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
                   "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out}/generations.jsonl")


if __name__ == "__main__":
    main()
