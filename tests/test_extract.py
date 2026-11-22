from gen.extract import extract

FN = "import pandas as pd\n\ndef generate_signals(df):\n    return df['close'] * 0\n"


def test_single_block():
    code, m = extract(f"Here you go:\n```python\n{FN}```\nThis works by ...")
    assert code == FN and m == "single_block"


def test_picks_block_with_function_and_last_draft():
    text = f"```python\nimport numpy as np\n```\nfirst try:\n```python\n{FN.replace('* 0', '* 1')}```\nfixed:\n```python\n{FN}```\n```python\nprint(generate_signals(df))\n```"
    code, m = extract(text)
    assert code == FN and m == "multi_block"


def test_unclosed_fence():
    code, m = extract(f"```python\n{FN}")
    assert code == FN and m == "unclosed"


def test_no_fence_trims_prose():
    code, m = extract(f"Sure! Below is the function.\n{FN}\nThis returns a series of positions.")
    assert code.strip() == FN.strip() and m == "no_fence"


def test_bare_fence_language():
    code, _ = extract(f"```\n{FN}```")
    assert code == FN
