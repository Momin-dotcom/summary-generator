import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "starter"))

from models import load_tokenizer
from data import load_sft_splits, load_preference_splits, load_eval_posts

tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)
pref_train, pref_val = load_preference_splits(tok)
eval_posts = load_eval_posts(tok)

p_len = np.array([len(x["prompt_ids"]) for x in sft_train])
s_len = np.array([len(x["summary_ids"]) for x in sft_train])

rows = [
    ("SFT train / validation pairs", f"{len(sft_train):,} / {len(sft_val):,}"),
    ("Preference train / validation pairs", f"{len(pref_train):,} / {len(pref_val):,}"),
    ("Evaluation posts", f"{len(eval_posts):,}"),
    ("Mean / max prompt length, SFT train (tokens)", f"{p_len.mean():.1f} / {p_len.max()}"),
    ("Mean / max summary length, SFT train (tokens)*", f"{s_len.mean():.1f} / {s_len.max()}"),
]

lines = ["| Item | Value |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in rows]
lines.append("")
lines.append("*Summary length includes the end-of-sequence token. "
             "Prompt length includes the `\\nTL;DR:` marker.")

table = "\n".join(lines)
print(table)

out = ROOT / "results"
out.mkdir(exist_ok=True)
(out / "table1.md").write_text(table, encoding="utf-8")