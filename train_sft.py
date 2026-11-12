import os, json, time
import torch
from functools import partial
from torch.utils.data import DataLoader
from common import token_logprobs, summary_mask, generate_ids, decode, peak_gb
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_eval_posts
from batching import collate_sft, collate_prompts
from config import SEED

BS, MB, LR, EVERY, PRINT_EVERY = 16, 4, 5e-5, 250, 25

torch.manual_seed(SEED)
t_start = time.time()
tok = load_tokenizer()
pad = tok.pad_token_id
train, val = load_sft_splits(tok)
eval_posts = load_eval_posts(tok)
print(f"data ready in {time.time() - t_start:.0f}s", flush=True)
model = load_causal_lm().float()
print("param dtype:", next(model.parameters()).dtype, flush=True)
dev = next(model.parameters()).device
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints/sft", exist_ok=True)


def samples(model, n=5):
    model.eval()
    items = eval_posts[:n]
    outs = generate_ids(model, collate_prompts(items, pad), tok)
    return [{"prompt": x["prompt"], "reference": x["summary"].strip(), "generated": decode(tok, o)}
            for x, o in zip(items, outs)]


@torch.no_grad()
def val_loss():
    model.eval()
    loader = DataLoader(val, 8, shuffle=False, collate_fn=partial(collate_sft, pad_id=pad))
    tot, n = 0.0, 0.0
    for b in loader:
        ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lp = token_logprobs(model, ids, mask)
        m = summary_mask(b["prompt_lens"], mask)
        tot += -(lp * m).sum().item()
        n += m.sum().item()
    model.train()
    return tot / n


before = samples(model)
print("before-SFT samples done", flush=True)
model.config.use_cache = False
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")
loader = DataLoader(train, BS, shuffle=True, collate_fn=partial(collate_sft, pad_id=pad))

log, run, t0 = [], 0.0, time.time()
for step, b in enumerate(loader, 1):
    ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
    m = summary_mask(b["prompt_lens"], mask)
    total = m.sum()
    opt.zero_grad(set_to_none=True)
    for i in range(0, ids.size(0), MB):
        sl = slice(i, i + MB)
        with torch.autocast("cuda", dtype=torch.float16):
            lp = token_logprobs(model, ids[sl], mask[sl])
        loss = -(lp * m[sl]).sum() / total
        scaler.scale(loss).backward()
        run += loss.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    if step % PRINT_EVERY == 0:
        print(f"step {step}/{len(loader)}  elapsed {time.time() - t0:.0f}s", flush=True)
    if step % EVERY == 0:
        rec = {"step": step, "train_loss": run / EVERY, "val_loss": val_loss()}
        run = 0.0
        log.append(rec)
        print(rec, flush=True)

elapsed = time.time() - t0
model.config.use_cache = True
after = samples(model)
model.save_pretrained("checkpoints/sft")
tok.save_pretrained("checkpoints/sft")
json.dump({"log": log, "time_sec": elapsed, "peak_gb": peak_gb(), "before": before, "after": after},
          open("results/sft_log.json", "w"), indent=2)
print("done", elapsed, peak_gb())