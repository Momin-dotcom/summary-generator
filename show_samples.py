import json

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl", encoding="utf-8")]
out = []
for m in ["ppo", "dpo"]:
    rs = sorted(rows, key=lambda r: r[m + "_score"] - r["sft_score"])
    for tag, sel in [("WORST", rs[:4]), ("BEST", rs[-4:])]:
        out.append(f"\n===== {m.upper()} {tag} =====")
        for r in sel:
            d = r[m + "_score"] - r["sft_score"]
            out.append(f"id {r['id']} | diff {d:+.2f}")
            out.append("POST END: " + r["prompt"][-250:].replace("\n", " "))
            out.append("REF: " + r["reference"])
            out.append("SFT: " + r["sft"])
            out.append("PPO: " + r["ppo"])
            out.append("DPO: " + r["dpo"])
            out.append("-" * 40)

text = "\n".join(out)
open("results/candidates.txt", "w", encoding="utf-8").write(text)
print(text)