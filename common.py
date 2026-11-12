import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import torch
import torch.nn.functional as F


def token_logprobs(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return -nll.view(targets.shape)


def summary_mask(prompt_lens, attention_mask):
    _, L = attention_mask.shape
    pos = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)
    pl = prompt_lens.to(attention_mask.device).unsqueeze(1)
    return ((pos >= pl) & (attention_mask[:, 1:] == 1)).float()


def build_batch(prompt_ids, summary_ids, pad_id, device):
    seqs = [p + s for p, s in zip(prompt_ids, summary_ids)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    pl = torch.tensor([len(p) for p in prompt_ids], dtype=torch.long)
    return ids.to(device), mask.to(device), pl.to(device)


@torch.no_grad()
def generate_ids(model, batch, tok, sample=False, max_new_tokens=48):
    dev = next(model.parameters()).device
    ids = batch["input_ids"].to(dev)
    mask = batch["attention_mask"].to(dev)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
              eos_token_id=tok.eos_token_id, use_cache=True)
    if sample:
        kw.update(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
    else:
        kw.update(do_sample=False)
    out = model.generate(input_ids=ids, attention_mask=mask, **kw)
    res = []
    for row in out[:, ids.size(1):].tolist():
        if tok.eos_token_id in row:
            row = row[: row.index(tok.eos_token_id) + 1]
        res.append(row)
    return res


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=True).strip()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0