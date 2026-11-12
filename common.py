import torch

def token_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()          # position t predicts token t+1
    targets = input_ids[:, 1:]
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return target_logit - torch.logsumexp(logits, dim=-1)

def summary_mask(attention_mask, prompt_lens):
    L = attention_mask.shape[1]
    pos = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)   # token index for each column
    after_prompt = pos >= prompt_lens.to(attention_mask.device).unsqueeze(1)
    is_real = attention_mask[:, 1:] == 1
    return (after_prompt & is_real).long()

def sft_loss(model, batch, device):
    ids = batch["input_ids"].to(device)
    am  = batch["attention_mask"].to(device)
    lp  = token_logprobs(model, ids, am)
    m   = summary_mask(am, batch["prompt_lens"]).float()
    n   = m.sum()
    return -(lp * m).sum() / n, n

@torch.no_grad()
def generate_greedy(model, tok, batch, max_new_tokens=48, device="cuda"):
    model.eval()
    ids = batch["input_ids"].to(device)
    am  = batch["attention_mask"].to(device)
    out = model.generate(input_ids=ids, attention_mask=am, do_sample=False,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    new = out[:, ids.shape[1]:]                 # only the newly generated part
    return [tok.decode(r, skip_special_tokens=True) for r in new]