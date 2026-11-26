import os, json, time, random, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import build_batch, generate_ids, summary_mask, token_logprobs, peak_gb
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_prompts
from config import SEED
from train_reward import load_reward_model

EPS, BETA, CV, CH = 0.2, 0.05, 0.1, 0.01
K, LR, NP, ITERS, MB = 4, 1e-5, 16, 200, 4
AC = dict(device_type="cuda", dtype=torch.float16)
DRIVE = "/content/drive/MyDrive/rlhf_ckpt"


def forward_all(policy, vhead, ids, mask):
    with torch.autocast(**AC):
        out = policy(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        v = vhead(out.hidden_states[-1][:, :-1]).squeeze(-1)
    logp_all = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    lp = logp_all.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1)
    return lp, v.float(), ent


@torch.no_grad()
def old_pass(policy, vhead, ids, mask, chunk=MB):
    lps, vs = [], []
    for i in range(0, ids.size(0), chunk):
        lp, v, _ = forward_all(policy, vhead, ids[i:i + chunk], mask[i:i + chunk])
        lps.append(lp)
        vs.append(v)
    return torch.cat(lps), torch.cat(vs)


@torch.no_grad()
def ref_pass(ref, ids, mask, chunk=MB):
    outs = []
    for i in range(0, ids.size(0), chunk):
        with torch.autocast(**AC):
            outs.append(token_logprobs(ref, ids[i:i + chunk], mask[i:i + chunk]))
    return torch.cat(outs)


@torch.no_grad()
def score_pass(rm, ids, mask, chunk=8):
    outs = []
    for i in range(0, ids.size(0), chunk):
        with torch.autocast(**AC):
            outs.append(rm(ids[i:i + chunk], mask[i:i + chunk]))
    return torch.cat(outs).float()


def backup():
    if not os.path.isdir(DRIVE):
        return
    try:
        shutil.copytree("checkpoints/ppo", os.path.join(DRIVE, "ppo"), dirs_exist_ok=True)
        shutil.copy("results/ppo_log.json", os.path.join(DRIVE, "ppo_log.json"))
        print("drive backup ok", flush=True)
    except Exception as e:
        print("drive backup skipped:", e, flush=True)


torch.manual_seed(SEED)
tok = load_tokenizer()
pad = tok.pad_token_id
sft_train, _ = load_sft_splits(tok)
policy = load_causal_lm("checkpoints/sft")
ref = load_causal_lm("checkpoints/sft")
rm = load_reward_model()
dev = next(policy.parameters()).device
vhead = nn.Linear(policy.config.hidden_size, 1).to(dev)
nn.init.zeros_(vhead.weight)
nn.init.zeros_(vhead.bias)

for mod in (ref, rm):
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
print("ref frozen:", all(not p.requires_grad for p in ref.parameters()))
print("reward frozen:", all(not p.requires_grad for p in rm.parameters()))

params = list(policy.parameters()) + list(vhead.parameters())
opt = torch.optim.AdamW(params, lr=LR)
scaler = torch.amp.GradScaler("cuda")
order = list(range(len(sft_train)))
random.Random(SEED).shuffle(order)
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints/ppo", exist_ok=True)


def save_policy():
    policy.save_pretrained("checkpoints/ppo")
    tok.save_pretrained("checkpoints/ppo")


log, t0 = [], time.time()
for it in range(ITERS):
    items = [sft_train[j] for j in order[it * NP:(it + 1) * NP]]
    batch = collate_prompts(items, pad)
    policy.eval()
    with torch.autocast(**AC):
        gen = generate_ids(policy, batch, tok, sample=True)
    ids, mask, pl = build_batch([x["prompt_ids"] for x in items], gen, pad, dev)
    m = summary_mask(pl, mask)
    B = ids.size(0)

    old_lp, old_v = old_pass(policy, vhead, ids, mask)
    ref_lp = ref_pass(ref, ids, mask)
    score = score_pass(rm, ids, mask)

    kl_tok = (old_lp - ref_lp) * m
    rew = -BETA * kl_tok
    pos = torch.arange(m.size(1), device=dev).unsqueeze(0)
    last = (m * (pos + 1)).argmax(dim=1)
    rew[torch.arange(B, device=dev), last] += score
    R = rew.flip(1).cumsum(1).flip(1) * m
    A = (R - old_v) * m
    vals = A[m > 0]
    A = ((A - vals.mean()) / (vals.std() + 1e-8)) * m
    N = m.sum()

    policy.train()
    clips, pgs, vls, ens = [], [], [], []
    ep0_clip, ep0_dev = 0.0, 0.0
    for ep in range(K):
        opt.zero_grad(set_to_none=True)
        pg_s = v_s = e_s = c_s = 0.0
        for i in range(0, B, MB):
            s = slice(i, i + MB)
            lp, v, ent = forward_all(policy, vhead, ids[s], mask[s])
            ms = m[s]
            ratio = torch.exp((lp - old_lp[s]) * ms)
            a = A[s]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - EPS, 1 + EPS) * a)
            pg_t = (surr * ms).sum() / N
            v_t = (((v - R[s]) ** 2) * ms).sum() / N
            e_t = (ent * ms).sum() / N
            loss = -pg_t + CV * v_t - CH * e_t
            scaler.scale(loss).backward()
            pg_s += -pg_t.item()
            v_s += v_t.item()
            e_s += e_t.item()
            c_s += (((ratio - 1).abs() > EPS).float() * ms).sum().item() / N.item()
            if ep == 0:
                ep0_dev = max(ep0_dev, ((ratio - 1).abs() * ms).max().item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        clips.append(c_s)
        pgs.append(pg_s)
        vls.append(v_s)
        ens.append(e_s)
        if ep == 0:
            ep0_clip = c_s

    rec = {"iter": it + 1,
           "score": score.mean().item(),
           "kl": kl_tok.sum(1).mean().item(),
           "length": m.sum(1).mean().item(),
           "clip_frac": sum(clips) / K,
           "clip_frac_epoch0": ep0_clip,
           "ratio_dev_epoch0": ep0_dev,
           "policy_loss": sum(pgs) / K,
           "value_loss": sum(vls) / K,
           "entropy": sum(ens) / K}
    log.append(rec)
    print(f"it {it + 1}/{ITERS} score {rec['score']:.3f} kl {rec['kl']:.3f} len {rec['length']:.1f} "
          f"clip {rec['clip_frac']:.3f} ep0_dev {ep0_dev:.2e} elapsed {time.time() - t0:.0f}s", flush=True)
    if it == 0:
        print("CHECK iter1: kl", rec["kl"], "ratio_dev_epoch0", ep0_dev, "clip_frac_epoch0", ep0_clip, flush=True)
    json.dump({"log": log, "time_sec": time.time() - t0, "peak_gb": peak_gb()},
              open("results/ppo_log.json", "w"), indent=2)
    if (it + 1) % 50 == 0:
        save_policy()
        backup()

save_policy()
backup()
print("done", time.time() - t0, peak_gb())
