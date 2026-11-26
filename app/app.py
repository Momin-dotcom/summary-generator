
import sys, os
sys.path.insert(0, os.getcwd())        
import torch, gradio as gr
from common import build_batch, generate_ids, decode   
from models import load_tokenizer, load_causal_lm
from data import clean_post
from train_reward import RewardModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = load_tokenizer(); pad = tok.pad_token_id

models = {n: load_causal_lm(f"checkpoints/{n.lower()}", DEV).eval() for n in ["SFT", "PPO", "DPO"]}
rm = RewardModel("checkpoints/sft")
rm.load_state_dict(torch.load("checkpoints/reward/reward.pt", map_location="cpu"))
rm = rm.to(DEV).eval()

@torch.no_grad()
def run(post):
    if not post or not post.strip():
        return ("Please paste a post.",) * 3
    p = tok(clean_post(post), add_special_tokens=False)["input_ids"]
    if len(p) > 448:
        return (f"Post too long ({len(p)} tokens, max 448).",) * 3
    batch = {"input_ids": torch.tensor([p]), "attention_mask": torch.ones(1, len(p), dtype=torch.long)}
    out = []
    for name, m in models.items():
        s = generate_ids(m, batch, tok, sample=False, max_new_tokens=48)[0]
        ids, mask, _ = build_batch([p], [s], pad, DEV)
        score = rm(ids, mask).item()
        out.append(f"{decode(tok, s)}\n\n[reward-model score: {score:.2f}]")
    return tuple(out)

demo = gr.Interface(
    run,
    gr.Textbox(lines=12, label="Reddit post (SUBREDDIT / TITLE / POST)"),
    [gr.Textbox(label=n) for n in models],
    title="SFT vs PPO vs DPO summaries",
)
if __name__ == "__main__":
    demo.launch(share=True)