"""Pull the solution code out of a model completion.

Handles: one fenced block (the normal case), several blocks (picks the one
that defines generate_signals), an unclosed fence (truncated output), and no
fences at all (trims surrounding prose until the code parses).

The extractor never edits the code itself: top-level example usage that the
model left in the block stays there and is scored as written.
"""

from __future__ import annotations

import ast
import re

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)
FUNC = "def generate_signals"


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _trim_prose(text: str) -> str:
    """No fences: start at the first code-looking line, drop trailing prose until it parses."""
    lines = text.strip().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"\s*(import |from |def |class |@|[A-Za-z_]\w*\s*=)", l)), 0)
    lines = lines[start:]
    for end in range(len(lines), 0, -1):
        code = "\n".join(lines[:end])
        if FUNC in code and _parses(code):
            return code
    return "\n".join(lines)


def extract(text: str) -> tuple[str, str]:
    """Return (code, method). method is one of: single_block, multi_block, unclosed, no_fence."""
    blocks = [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text)]
    closed = text.count("```") % 2 == 0
    if not blocks:
        return _trim_prose(text).strip() + "\n", "no_fence"
    if len(blocks) == 1:
        return blocks[0][1].strip() + "\n", "single_block" if closed else "unclosed"
    with_fn = [b for _, b in blocks if FUNC in b]
    if with_fn:
        # several blocks define it (e.g. a draft then a fix): the last one is the final answer
        code = with_fn[-1]
    else:
        py = [b for lang, b in blocks if lang in ("python", "py", "")]
        code = max(py or [b for _, b in blocks], key=len)
    return code.strip() + "\n", "multi_block"
