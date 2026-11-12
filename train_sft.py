import json, os, random, sys
from functools import partial
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

from common import sft_loss, generate_greedy
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_eval_posts
from batching import collate_sft, collate_prompts
from config import SEED

LR, BATCH, CLIP, LOG_EVERY = 5e-5, 16, 1.0, 250
os.makedirs("results", exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)
eval_posts = load_eval_posts(tok)
pad = tok.pad_token_id

train_dl = DataLoader(sft_train, BATCH, shuffle=True, collate_fn=partial(collate_sft, pad_id=pad))
val_dl   = DataLoader(sft_val, 32, shuffle=False,   collate_fn=partial(collate_sft, pad_id=pad))

gen_posts = eval_posts[:5]
gen_batch = collate_prompts(gen_posts, pad)

model = load_causal_lm(device=device)

@torch.no_grad()
def validate():
    model.eval()
    tot, cnt = 0.0, 0.0
    for b in val_dl:
        loss, n = sft_loss(model, b, device)
        tot += loss.item() * n.item()           # token-weighted
        cnt += n.item()
    model.train()
    return tot / cnt

# 2.5 BEFORE SFT
before = generate_greedy(model, tok, gen_batch, 48, device)

# 2.3 training: 1 epoch = 20000/16 = 1250 steps
opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
log, window, step = [], [], 0
log.append({"step": 0, "train_loss": None, "val_loss": validate()})

for batch in train_dl:
    loss, _ = sft_loss(model, batch, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    step += 1
    window.append(loss.item())
    if step % LOG_EVERY == 0:                   # 2.4
        tr, va = float(np.mean(window)), validate()
        window = []
        log.append({"step": step, "train_loss": tr, "val_loss": va})
        print(f"step {step} | train {tr:.4f} | val {va:.4f}")

# 2.4 save
os.makedirs("checkpoints/sft", exist_ok=True)
model.save_pretrained("checkpoints/sft")
tok.save_pretrained("checkpoints/sft")

# 2.5 AFTER SFT
after = generate_greedy(model, tok, gen_batch, 48, device)
samples = [{"prompt_tail": p["prompt"][-400:], "reference": p["summary"],
            "before_sft": b, "after_sft": a}
           for p, b, a in zip(gen_posts, before, after)]
json.dump(samples, open("results/sft_samples.json", "w"), indent=2)
json.dump(log,     open("results/sft_log.json", "w"), indent=2)

# loss figure
tr = [(l["step"], l["train_loss"]) for l in log if l["train_loss"] is not None]
va = [(l["step"], l["val_loss"]) for l in log]
plt.figure(figsize=(6, 4))
plt.plot(*zip(*tr), marker="o", label="train")
plt.plot(*zip(*va), marker="s", label="validation")
plt.xlabel("step"); plt.ylabel("NLL per summary token"); plt.legend(); plt.tight_layout()
plt.savefig("results/fig1_sft_loss.png", dpi=150)