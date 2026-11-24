import os
import json
import time
import shutil
import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import token_logprobs, summary_mask, peak_gb
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref

# Task 5 fixed settings
BETA = 0.1
LR = 2e-6
BATCH_SIZE = 8        # 8 preference pairs per optimizer step
MB = 4                # microbatch size (4 pairs x 2 accum steps = exact batch 8, safe on T4)
GRAD_ACCUM = BATCH_SIZE // MB
GRAD_CLIP = 1.0
LOG_EVERY = 250
BACKUP_EVERY = 1000
SEED = 42

SFT_DIR = "checkpoints/sft"
OUT_DIR = "checkpoints/dpo"
LOG_PATH = "results/dpo_log.json"
DRIVE = "/content/drive/MyDrive/rlhf_ckpt"


def seq_logprobs(model, input_ids, attention_mask, prompt_lens):
    """Sum of token log-probabilities over summary tokens only (Task 5.2)."""
    lp = token_logprobs(model, input_ids, attention_mask)     # (B, L-1), fp32
    m = summary_mask(prompt_lens, attention_mask)             # (B, L-1), fp32
    return (lp * m).sum(dim=1)                                # (B,)


def dpo_margins(policy, ref, batch, device):
    ch_ids = batch["chosen_ids"].to(device)
    ch_mask = batch["chosen_mask"].to(device)
    rj_ids = batch["rejected_ids"].to(device)
    rj_mask = batch["rejected_mask"].to(device)
    pl = batch["prompt_lens"].to(device)
    use_amp = device.startswith("cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        ref_ch = seq_logprobs(ref, ch_ids, ch_mask, pl)
        ref_rj = seq_logprobs(ref, rj_ids, rj_mask, pl)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        pol_ch = seq_logprobs(policy, ch_ids, ch_mask, pl)
        pol_rj = seq_logprobs(policy, rj_ids, rj_mask, pl)

    return BETA * ((pol_ch - ref_ch) - (pol_rj - ref_rj))


@torch.no_grad()
def evaluate(policy, ref, val_loader, device):
    policy.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in val_loader:
        margins = dpo_margins(policy, ref, batch, device)
        loss = -F.logsigmoid(margins)
        total_loss += loss.sum().item()
        correct += (margins > 0).sum().item()
        total += margins.size(0)
    policy.train()
    return total_loss / total, correct / total


def save_log(summary, also_drive=True):
    with open(LOG_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    if also_drive and os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE, exist_ok=True)
        shutil.copy2(LOG_PATH, os.path.join(DRIVE, "dpo_log.json"))


def backup_checkpoint_to_drive():
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE, exist_ok=True)
        dst = os.path.join(DRIVE, "dpo")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(OUT_DIR, dst)
        print("drive backup ok")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)
    collate = partial(collate_pref, pad_id=tok.pad_token_id)

    train_loader = DataLoader(pref_train, batch_size=MB, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(pref_val, batch_size=8, shuffle=False, collate_fn=collate)

    # Task 5.1: load checkpoints/sft/ twice (trainable policy + frozen reference)
    policy = load_causal_lm(SFT_DIR, device=device)
    ref = load_causal_lm(SFT_DIR, device=device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # Correctness check (Section 3.3): frozen reference model
    ref_frozen = all(not p.requires_grad for p in ref.parameters())
    pol_trainable = all(p.requires_grad for p in policy.parameters())
    print(f"CHECK ref frozen: {ref_frozen} | policy trainable: {pol_trainable}")
    assert ref_frozen, "Reference model must be frozen!"

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    t0 = time.time()
    policy.train()
    opt.zero_grad(set_to_none=True)

    log = []
    step = 0
    step1_loss = None
    running_loss = 0.0
    running_steps = 0
    micro_loss = 0.0
    micro_count = 0

    for batch in train_loader:
        margins = dpo_margins(policy, ref, batch, device)
        loss = -F.logsigmoid(margins).mean()

        scaler.scale(loss / GRAD_ACCUM).backward()
        micro_loss += loss.item()
        micro_count += 1

        if micro_count == GRAD_ACCUM:
            step_loss = micro_loss / GRAD_ACCUM
            step += 1

            # Correctness check (Section 3.3): step 1 loss == log(2) ~= 0.6931
            if step == 1:
                step1_loss = step_loss
                print(f"CHECK step 1 loss: {step1_loss:.6f} (expected log 2 = 0.693147)")

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            running_loss += step_loss
            running_steps += 1
            micro_loss = 0.0
            micro_count = 0

            if step % LOG_EVERY == 0:
                avg_train_loss = running_loss / running_steps
                val_loss, val_acc = evaluate(policy, ref, val_loader, device)
                elapsed = time.time() - t0
                entry = {
                    "step": step,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "elapsed_sec": elapsed,
                }
                log.append(entry)
                print(
                    f"step {step:4d} | train_loss {avg_train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {elapsed:.1f}s"
                )
                running_loss = 0.0
                running_steps = 0

                # Mid-run log backup to Drive
                save_log({
                    "beta": BETA,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "step1_loss": step1_loss,
                    "log": log,
                })

                # Mid-run checkpoint backup to Drive
                if step % BACKUP_EVERY == 0:
                    policy.save_pretrained(OUT_DIR)
                    tok.save_pretrained(OUT_DIR)
                    backup_checkpoint_to_drive()

    # Final save and backup
    policy.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    total_time = time.time() - t0
    peak = peak_gb()
    summary = {
        "beta": BETA,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "step1_loss": step1_loss,
        "final_train_loss": log[-1]["train_loss"],
        "final_val_loss": log[-1]["val_loss"],
        "final_val_acc": log[-1]["val_acc"],
        "time_sec": total_time,
        "peak_gb": peak,
        "log": log,
    }
    save_log(summary)
    backup_checkpoint_to_drive()
    print(f"done {total_time:.1f}s | peak {peak:.2f} GB | final_val_acc {log[-1]['val_acc']:.4f}")


if __name__ == "__main__":
    main()