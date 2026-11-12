import sys
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "starter"))
sys.path.append(str(ROOT))

from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_sft
from common import token_logprobs, summary_mask

tok = load_tokenizer()
sft_train, _ = load_sft_splits(tok)
model = load_causal_lm()
model.eval()
device = next(model.parameters()).device

loader = DataLoader(sft_train, batch_size=4, shuffle=False,
                    collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))
b = next(iter(loader))
ids = b["input_ids"].to(device)
am = b["attention_mask"].to(device)

with torch.no_grad():
    lp = token_logprobs(model, ids, am)
mask = summary_mask(am, b["prompt_lens"])

print("logprobs shape:", tuple(lp.shape), "mask shape:", tuple(mask.shape))
assert lp.shape == mask.shape == (ids.shape[0], ids.shape[1] - 1)
assert (lp <= 0).all(), "log-probs must be <= 0"

all_ok = True
for i in range(ids.shape[0]):
    picked = ids[i, 1:][mask[i].bool()].tolist()
    decoded = tok.decode(picked)
    expected = sft_train[i]["summary"] + tok.eos_token
    ok = decoded == expected
    all_ok &= ok
    print(f"\nrow {i}: match = {ok}")
    print("decoded :", repr(decoded))
    print("expected:", repr(expected))

print("\nALL ROWS MATCH:", all_ok)