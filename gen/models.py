"""Model registry. Fixed once Phase 2 starts.

The Llama 3.1 8B 4-bit weights are shared, read-only, with the separate
local-rag-10k-eval project instead of being downloaded a second time.
Override with LLAMA_8B_PATH if they live elsewhere.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_8B = os.environ.get("LLAMA_8B_PATH", str(ROOT.parent / "llama-project" / "llama-mlx-quantized"))

MODELS = {
    "qwen3b": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",    # fine-tuning target
    "qwen1.5b": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",  # small reference
    "qwen7b": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",    # larger reference, no fine-tune
    "llama8b": LLAMA_8B,                                         # general model, 4-bit
}
