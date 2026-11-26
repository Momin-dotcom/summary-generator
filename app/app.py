# app/app.py   (run: python app/app.py from the repo root)
import sys; sys.path.extend(["starter", "."])
import torch, gradio as gr
from models import load_tokenizer, load_causal_lm
from data import clean_post
from train_reward import RewardModel
from common import *

tok = load_tokenizer()
models = {n: load_causal_lm(p, DEV).eval() for n, p in
          {"SFT": "checkpoints/sft", "PPO": "checkpoints/ppo", "DPO": "checkpoints/dpo"}.items()}
rm = RewardModel(); rm.load_state_dict(torch.load("checkpoints/reward/reward.pt", map_location="cpu"))
rm = rm.to(DEV).eval()

@torch.no_grad()
def run(post):
    p = tok(clean_post(post), add_special_tokens=False)["input_ids"]
    if len(p) > 448: return ("Post too long (max 448 tokens).",) * 3
    out = []
    for name, m in models.items():
        s = generate_greedy(m, tok, [{"prompt_ids": p}])[0]
        ids, mask, _ = make_batch([p], [s], tok.pad_token_id)
        out.append(f"{dec(tok, s)}\n\n[reward-model score: {rm(ids, mask).item():.2f}]")
    return tuple(out)

gr.Interface(run, gr.Textbox(lines=12, label="Reddit post (SUBREDDIT / TITLE / POST)"),
             [gr.Textbox(label=n) for n in models], title="SFT vs PPO vs DPO summaries").launch(share=True)