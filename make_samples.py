import json
rows = {r["id"]: r for r in map(json.loads, open("results/eval_summaries.jsonl", encoding="utf-8"))}

picks = [
 (207, "BETTER (PPO and DPO)", "SFT is vague ('not sure how to make her feel comfortable'). PPO and DPO capture the real question: connecting on a deeper level with an introverted girl."),
 (21,  "BETTER (PPO)", "SFT falls into a repetition loop. PPO states the actual issue: conflicting feelings depending on whether he is with her. Closest to the reference."),
 (87,  "BETTER (PPO, DPO)", "SFT repeats one sentence twice and misses the point. PPO is short but correct (a man won't leave her alone at lunch). DPO adds the 'what do I do' question."),
 (113, "BETTER (PPO and DPO)", "SFT is generic. Both capture the money question (what happens to the $ paid on medical leave). DPO also gives the reason (anxiety, missed classes)."),
 (174, "WORSE (PPO)", "PPO copies a chunk of the post and is cut off mid-sentence at the 48-token limit ('...a girl with a'). Lower rφ than SFT matches what we see."),
 (160, "WORSE (PPO)", "PPO keeps only 'rough New Year' and 'SO is not talking to me' and drops the request for help. SFT is more complete."),
 (41,  "WORSE (PPO, DPO)", "PPO output is just a Reddit-style header ('Me [23/M] started seeing a new girl [26/F]') with no problem or question. DPO copies the post and is truncated."),
 (277, "WORSE (PPO, DPO)", "PPO uses a 'TIFU' format and drops the outcome. DPO ends with 'because I told her I really liked her', which does not match what the post says (honest about feelings for his ex)."),
 (135, "SCORE HIGH, CONTENT SUSPECT", "rφ rises by +4.3 (PPO) and +4.95 (DPO) over SFT, and SFT loops. But PPO/DPO both say the girlfriend 'wants a kid', which the reference (NYE with friends vs her) does not mention. Check the full post. If it is not there, this is a fluent summary the reward model likes but that is factually wrong."),
 (26,  "WORSE (DPO)", "DPO is cut off, repeats $1098.90 and says a refund happened. Check the post for whether that is supported. Lower rφ than SFT by 2.54."),
]

with open("results/samples.md", "w", encoding="utf-8") as f:
    f.write("# Qualitative samples (10 evaluation posts)\n\n")
    for i, (pid, label, why) in enumerate(picks, 1):
        r = rows[pid]
        f.write(f"## {i}. Post id {pid} - {label}\n\n")
        f.write(f"**Post (end):** ...{r['prompt'][-400:].strip()}\n\n")
        f.write(f"- **Human reference:** {r['reference']}\n")
        f.write(f"- **SFT** (rφ {r['sft_score']:.2f}): {r['sft']}\n")
        f.write(f"- **PPO** (rφ {r['ppo_score']:.2f}): {r['ppo']}\n")
        f.write(f"- **DPO** (rφ {r['dpo_score']:.2f}): {r['dpo']}\n\n")
        f.write(f"**Why:** {why}\n\n---\n\n")
print("wrote results/samples.md")