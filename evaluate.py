import json, csv, random
import torch
from rouge_score import rouge_scorer
from common import token_logprobs, summary_mask, build_batch, generate_ids, decode
from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts
from train_reward import RewardModel         

dev = "cuda"
tok = load_tokenizer(); pad = tok.pad_token_id
posts = load_eval_posts(tok)
N = len(posts); print("eval posts:", N)

pols = {n: load_causal_lm(f"checkpoints/{n}").eval() for n in ["sft", "ppo", "dpo"]}

def load_rm():
    rm = RewardModel("checkpoints/sft")
    rm.load_state_dict(torch.load("checkpoints/reward/reward.pt", map_location="cpu"))
    return rm.to(dev).eval()
rm = load_rm()

@torch.no_grad()
def rm_score(ids, mask):
    return rm(ids, mask)                     

BS = 16
def chunks():
    for i in range(0, N, BS):
        yield i, min(i + BS, N)

def greedy(model):
    out = []
    for a, b in chunks():
        batch = collate_prompts(posts[a:b], pad)
        out += generate_ids(model, batch, tok, sample=False, max_new_tokens=48)
    return out

@torch.no_grad()
def scores(sums, extra_pad=0):
    out = []
    for a, b in chunks():
        ids, mask, _ = build_batch([p["prompt_ids"] for p in posts[a:b]], sums[a:b], pad, dev)
        if extra_pad:
            ids = torch.cat([ids, torch.full((len(ids), extra_pad), pad, device=dev)], 1)
            mask = torch.cat([mask, torch.zeros(len(ids), extra_pad, dtype=mask.dtype, device=dev)], 1)
        out += rm_score(ids, mask).tolist()
    return out

@torch.no_grad()
def seq_logp(model, sums):
    out = []
    for a, b in chunks():
        ids, mask, pl = build_batch([p["prompt_ids"] for p in posts[a:b]], sums[a:b], pad, dev)
        lp = token_logprobs(model, ids, mask) * summary_mask(pl, mask)
        out += lp.sum(1).tolist()
    return out

summ  = {n: greedy(m) for n, m in pols.items()}
score = {n: scores(s) for n, s in summ.items()}
lp_own = {n: seq_logp(pols[n], summ[n]) for n in pols}
lp_sft = {n: seq_logp(pols["sft"], summ[n]) for n in pols}   # SFT = reference

rs = rouge_scorer.RougeScorer(["rougeL"])
refs = [p["summary"].strip() for p in posts]
mean = lambda x: sum(x) / len(x)

metrics = {}
for n in pols:
    metrics[n] = {
        "mean_rm": mean(score[n]),
        "win_vs_sft": None if n == "sft" else mean([a > b for a, b in zip(score[n], score["sft"])]),
        "kl_to_sft": mean([a - b for a, b in zip(lp_own[n], lp_sft[n])]),
        "length": mean([len(s) for s in summ[n]]),
        "rougeL": mean([rs.score(r, decode(tok, s))["rougeL"].fmeasure for r, s in zip(refs, summ[n])]),
    }
json.dump(metrics, open("results/eval_metrics.json", "w"), indent=2)
with open("results/table3.md", "w") as f:
    f.write("| Policy | Mean rφ | Win rate vs SFT | KL to SFT | Length | ROUGE-L |\n|---|---|---|---|---|---|\n")
    for n, m in metrics.items():
        w = "–" if m["win_vs_sft"] is None else f"{m['win_vs_sft']:.3f}"
        f.write(f"| {n.upper()} | {m['mean_rm']:.3f} | {w} | {m['kl_to_sft']:.3f} | {m['length']:.1f} | {m['rougeL']:.3f} |\n")
print(open("results/table3.md").read())


with open("results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
    for i, p in enumerate(posts):
        row = {"id": i, "prompt": p["prompt"], "reference": refs[i]}
        for n in pols:
            row[n] = decode(tok, summ[n][i]); row[n + "_score"] = score[n][i]
        f.write(json.dumps(row) + "\n")


a = scores(summ["sft"]); b = scores(summ["sft"], extra_pad=20)
diff = max(abs(x - y) for x, y in zip(a, b))
print("PADDING CHECK max |diff| =", diff)
open("results/check_padding_output.txt", "w").write(f"max |diff| = {diff}\n")


rng = random.Random(0)
chosen = rng.sample(range(N), 30)
key, rows = {}, []
for i in chosen:
    order = ["sft", "ppo", "dpo"]; rng.shuffle(order)
    key[i] = dict(zip("ABC", order))
    rows.append([i, posts[i]["prompt"][-1200:]] + [decode(tok, summ[n][i]) for n in order] + ["", ""])
with open("results/human_eval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["post_id", "post", "A", "B", "C", "member1_pick", "member2_pick"])
    w.writerows(rows)
json.dump(key, open("results/human_eval_key.json", "w"), indent=1)