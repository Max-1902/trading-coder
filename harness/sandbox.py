"""Static checks and sandboxed execution of generated strategy code.

Layers:
1. AST pass: only numpy / pandas / math imports, no dangerous builtins,
   no dunder attribute access, no pandas/numpy file I/O, and a top-level
   `generate_signals` must exist.
2. Subprocess (`python -I`) under macOS `sandbox-exec` with network and
   file writes denied, killed after a wall-clock timeout.

This is a benchmark sandbox against accidental damage by a small model, not
a security boundary against a determined attacker.
"""

from __future__ import annotations

import ast
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "getattr", "setattr", "delattr",
    "memoryview", "help",
}
BANNED_ATTR_PREFIXES = ("read_", "to_csv", "to_pickle", "to_parquet", "to_excel",
                        "to_json", "to_hdf", "to_sql", "to_feather", "tofile")
BANNED_ATTRS = {"load", "save", "savez", "savetxt", "loadtxt", "fromfile", "genfromtxt", "system"}
FUNC_NAME = "generate_signals"

RUNNER = Path(__file__).with_name("_runner.py")
SANDBOX_PROFILE = (
    "(version 1)(allow default)(deny network*)"
    "(deny file-write*)(allow file-write* (literal \"/dev/null\"))"
)
DEFAULT_TIMEOUT_S = 30.0


@dataclass
class StaticResult:
    ok: bool
    category: str | None = None  # syntax_error | banned_import | banned_call | no_function
    detail: str = ""


def static_check(code: str) -> StaticResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticResult(False, "syntax_error", f"line {e.lineno}: {e.msg}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return StaticResult(False, "banned_import", alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in ALLOWED_IMPORTS:
                return StaticResult(False, "banned_import", "." * node.level + mod)
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return StaticResult(False, "banned_call", node.id)
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") or a in BANNED_ATTRS or a.startswith(BANNED_ATTR_PREFIXES):
                return StaticResult(False, "banned_call", f".{a}")

    has_fn = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == FUNC_NAME
        for n in tree.body
    )
    if not has_fn:
        return StaticResult(False, "no_function", f"no top-level {FUNC_NAME}()")
    return StaticResult(True)


@dataclass
class RunResult:
    status: str  # ok | timeout | crash | module_error
    detail: str = ""
    # one entry per job: {"ok": True, "pos": [...]} or {"ok": False, "error": cat, "detail": str}
    outputs: dict = field(default_factory=dict)
    la_cuts: dict = field(default_factory=dict)  # series -> list of [k, mode] actually run
    seconds: float = 0.0


def run_in_sandbox(code: str, series: dict, lookahead: dict | None = None,
                   timeout: float = DEFAULT_TIMEOUT_S) -> RunResult:
    """Run generate_signals on every series (and look-ahead variants) in one subprocess.

    lookahead: {"series": [names], "n_random": int, "n_transition": int,
                "modes": [...], "seed": int}
    """
    import time

    payload = pickle.dumps({"code": code, "series": series, "lookahead": lookahead,
                            "project_root": str(RUNNER.parent.parent)})
    cmd = [sys.executable, "-I", "-B", str(RUNNER)]
    if shutil.which("sandbox-exec"):
        cmd = ["sandbox-exec", "-p", SANDBOX_PROFILE, *cmd]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult("timeout", f">{timeout:.0f}s", seconds=time.perf_counter() - t0)
    secs = time.perf_counter() - t0

    lines = proc.stdout.decode(errors="replace").strip().splitlines()
    if proc.returncode != 0 or not lines:
        err = proc.stderr.decode(errors="replace")[-2000:]
        return RunResult("crash", f"exit {proc.returncode}: {err}", seconds=secs)
    try:
        msg = json.loads(lines[-1])
    except json.JSONDecodeError:
        return RunResult("crash", "runner produced no result line", seconds=secs)
    if msg.get("module_error"):
        return RunResult("module_error", msg["module_error"], seconds=secs)
    return RunResult("ok", outputs=msg["outputs"], la_cuts=msg.get("la_cuts", {}), seconds=secs)
