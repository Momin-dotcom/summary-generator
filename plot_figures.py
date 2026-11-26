import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
L = lambda f: json.load(open(f"{R}/{f}.json"))
sft, rew, dpo, ppo = L("sft_log"), L("reward_log"), L("dpo_log"), L("ppo_log")

s = sft["log"]
plt.figure(figsize=(6, 4))
plt.plot([e["step"] for e in s], [e["train_loss"] for e in s], marker="o", label="train")
plt.plot([e["step"] for e in s], [e["val_loss"] for e in s], marker="s", label="validation")
plt.xlabel("step"); plt.ylabel("loss"); plt.title("SFT loss"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/fig1_sft.png", dpi=150); plt.close()


r = rew["log"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([e["step"] for e in r], [e["train_loss"] for e in r], marker="o")
ax[0].set_title("Reward model: train loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
ax[1].plot([e["step"] for e in r], [e["val_acc"] for e in r], marker="o", color="tab:green")
ax[1].axhline(0.5, ls="--", color="gray", label="chance")
ax[1].set_title("Reward model: validation accuracy"); ax[1].set_xlabel("step"); ax[1].set_ylabel("accuracy")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/fig2_reward.png", dpi=150); plt.close()


p = ppo["log"]; it = [e["iter"] for e in p]
def smooth(y, k=10):
    return [sum(y[max(0, i - k + 1): i + 1]) / len(y[max(0, i - k + 1): i + 1]) for i in range(len(y))]
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for a, key, title in zip(ax.flat, ["score", "kl", "length", "clip_frac"],
                         ["Mean reward-model score", "Mean KL to reference", "Mean summary length (tokens)", "Clip fraction"]):
    y = [e[key] for e in p]
    a.plot(it, y, alpha=.35, label="raw"); a.plot(it, smooth(y), label="moving avg (10)")
    a.set_title(title); a.set_xlabel("iteration"); a.grid(alpha=.3)
ax[0, 0].legend()
plt.tight_layout(); plt.savefig(f"{R}/fig3_ppo.png", dpi=150); plt.close()

d = dpo["log"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([e["step"] for e in d], [e["train_loss"] for e in d], marker="o", label="train")
ax[0].plot([e["step"] for e in d], [e["val_loss"] for e in d], marker="s", label="validation")
ax[0].set_title("DPO loss"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot([e["step"] for e in d], [e["val_acc"] for e in d], marker="o", color="tab:green")
ax[1].axhline(0.5, ls="--", color="gray", label="chance")
ax[1].set_title("DPO: validation implicit reward accuracy"); ax[1].set_xlabel("step")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/fig4_dpo.png", dpi=150); plt.close()


print("SFT    : val loss", round(sft["log"][-1]["val_loss"], 4), "| train loss", round(sft["log"][-1]["train_loss"], 4),
      "| time", round(sft["time_sec"]), "s | peak", round(sft["peak_gb"], 2), "GB")
print("Reward : val acc", round(rew["final_val_acc"], 4), "| train loss", round(rew["log"][-1]["train_loss"], 4),
      "| time", round(rew["time_sec"]), "s | peak", round(rew["peak_gb"], 2), "GB")
print("PPO    : final mean score (last 10 it)", round(sum(e["score"] for e in p[-10:]) / 10, 3),
      "| time", round(ppo["time_sec"]), "s | peak", round(ppo["peak_gb"], 2), "GB")
print("DPO    : val acc", round(dpo["final_val_acc"], 4), "| train loss", round(dpo["final_train_loss"], 4),
      "| time", round(dpo["time_sec"]), "s | peak", round(dpo["peak_gb"], 2), "GB")
print("PPO final KL (last 10 it):", round(sum(e["kl"] for e in p[-10:]) / 10, 3),
      "| max clip_frac:", round(max(e["clip_frac"] for e in p), 3))