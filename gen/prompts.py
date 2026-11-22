"""Chat messages for zero-shot and k-shot prompting.

Few-shot exemplars are fixed training tasks (leakage-checked against the test
set) shown as earlier user/assistant turns with the reference solution as the
answer. The same exemplars are used for every task and every model.
"""

from __future__ import annotations

from functools import lru_cache

FEWSHOT_FAMILIES = ["ma_cross", "oscillator", "stop_tp"]  # stateless, hysteresis, stateful


@lru_cache(maxsize=1)
def fewshot_examples() -> list[dict]:
    from gen.make_train import generate

    train = generate(per_family=20, seed=0)
    out = []
    for fam in FEWSHOT_FAMILIES:
        # first long/short task of the family, so exemplars show both sides
        t = next(t for t in train if t["family"] == fam and t["params"].get("side") == "long_short")
        out.append(t)
    return out


def answer(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def messages(prompt: str, shots: int = 0) -> list[dict]:
    msgs = []
    for ex in fewshot_examples()[:shots]:
        msgs += [{"role": "user", "content": ex["prompt"]},
                 {"role": "assistant", "content": answer(ex["reference"])}]
    msgs.append({"role": "user", "content": prompt})
    return msgs
