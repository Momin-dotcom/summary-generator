import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from common import peak_gb
from models import load_tokenizer
from data import load_preference_splits
from batching import collate_pref
from config import SEED


class RewardModel(nn.Module):
    def __init__(self, base_path="checkpoints/sft"):
        super().__init__()
        lm = AutoModelForCausalLM.from_pretrained(base_path)
        self.body = lm.model
        self.head = nn.Linear(lm.config.hidden_size, 1)
        del lm

    def forward(self, input_ids, attention_mask):
        h = self.body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        last = (attention_mask * pos).argmax(dim=1)
        pooled = h[torch.arange(input_ids.size(0), device=input_ids.device), last]
        return self.head(pooled).squeeze(-1).float()


def load_reward_model(path="checkpoints/reward", base="checkpoints/sft", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel(base)
    model.load_state_dict(torch.load(os.path.join(path, "reward.pt"), map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def pair_scores(model, data, pad, dev, bs=16):
    model.eval()
    loader = DataLoader(data, bs, shuffle=False, collate_fn=partial(collate_pref, pad_id=pad))
    rcs, rrs = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            rcs.append(model(b["chosen_ids"].to(dev), b["chosen_mask"].to(dev)).cpu())
            rrs.append(model(b["rejected_ids"].to(dev), b["rejected_mask"].to(dev)).cpu())
    model.train()
    return torch.cat(rcs), torch.cat(rrs)


if __name__ == "__main__":
    BS, MB, LR, EVERY, PRINT_EVERY = 8, 4, 1e-5, 250, 50
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    pad = tok.pad_token_id
    train, val = load_preference_splits(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardModel("checkpoints/sft").to(dev)
    model.body.config.use_cache = False
    model.train()
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/reward", exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(train, BS, shuffle=True, collate_fn=partial(collate_pref, pad_id=pad))
    print("data ready, steps:", len(loader), flush=True)

    log, run, t0 = [], 0.0, time.time()
    for step, b in enumerate(loader, 1):
        B = b["chosen_ids"].size(0)
        opt.zero_grad(set_to_none=True)
        for i in range(0, B, MB):
            sl = slice(i, i + MB)
            with torch.autocast("cuda", dtype=torch.float16):
                rc = model(b["chosen_ids"][sl].to(dev), b["chosen_mask"][sl].to(dev))
                rr = model(b["rejected_ids"][sl].to(dev), b["rejected_mask"][sl].to(dev))
            loss = -F.logsigmoid(rc - rr).sum() / B
            scaler.scale(loss).backward()
            run += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % PRINT_EVERY == 0:
            print(f"step {step}/{len(loader)}  elapsed {time.time() - t0:.0f}s", flush=True)
        if step % EVERY == 0:
            rc_v, rr_v = pair_scores(model, val, pad, dev)
            rec = {"step": step, "train_loss": run / EVERY, "val_acc": (rc_v > rr_v).float().mean().item()}
            run = 0.0
            log.append(rec)
            print(rec, flush=True)

    elapsed = time.time() - t0
    rc_v, rr_v = pair_scores(model, val, pad, dev)
    acc = (rc_v > rr_v).float().mean().item()
    wrong = []
    for i in (rc_v <= rr_v).nonzero().squeeze(-1).tolist()[:30]:
        wrong.append({"prompt": val[i]["prompt"], "chosen": val[i]["chosen"], "rejected": val[i]["rejected"],
                      "r_chosen": rc_v[i].item(), "r_rejected": rr_v[i].item()})
    model.body.config.use_cache = True
    torch.save(model.state_dict(), "checkpoints/reward/reward.pt")
    tok.save_pretrained("checkpoints/reward")
    json.dump({"log": log, "final_val_acc": acc, "time_sec": elapsed, "peak_gb": peak_gb()},
              open("results/reward_log.json", "w"), indent=2)
    json.dump(wrong, open("results/reward_errors.json", "w"), indent=2)
    print("final val acc", acc, "time", elapsed, "peak_gb", peak_gb())