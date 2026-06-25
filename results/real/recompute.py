"""recompute.py — reproduce analysis_summary.txt from per-example data alone (stdlib).
Reads eval_points.jsonl ({section, eval_order, pred, gold}); each section's accuracy
= mean(pred==gold) with seeded bootstrap 95% CI.
Gate: cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty)."""
import json, random
from collections import defaultdict
BOOT_N, BOOT_SEED, CHANCE = 2000, 42, 0.25
SECTIONS = [
    ("Base model: MMLU accuracy (argmax of option logits)", "base"),
    ("RL model (GRPO): MMLU accuracy", "rl"),
    ("Global affine fit to RL, applied to base: MMLU accuracy", "affine_to_rl"),
    ("Global affine fit to gold (5-fold OOF): MMLU accuracy", "affine_to_gold"),
    ("Full linear map fit to gold (5-fold OOF, ceiling): MMLU accuracy", "full_linear_gold"),
]
def pct(s,q):
    if not s: return float("nan")
    p=q/100.0*(len(s)-1); lo=int(p); f=p-lo
    return s[lo]*(1-f)+s[lo+1]*f if lo+1<len(s) else s[lo]
def acc_ci(preds,golds):
    n=len(preds); point=sum(1 for p,g in zip(preds,golds) if p==g)/n if n else float("nan")
    rng=random.Random(BOOT_SEED); boots=[]
    for _ in range(BOOT_N):
        c=sum(1 for _ in range(n) if (lambda i: preds[i]==golds[i])(rng.randrange(n)))
        boots.append(c/n)
    boots.sort(); return point, pct(boots,2.5), pct(boots,97.5), n
def main():
    by=defaultdict(list)
    for line in open("eval_points.jsonl"):
        line=line.strip()
        if line: r=json.loads(line); by[r["section"]].append(r)
    for k in by: by[k].sort(key=lambda r: r["eval_order"])
    L=["# Global affine logit transform vs GRPO: a scalar elicitation test",
       "","Model: Qwen2.5-1.5B-Instruct base (checkpoint-0) vs outcome-only GRPO (checkpoint-final), MMLU.",
       "A global affine transform is a single scalar temperature plus a per-option bias on the 4 option logits.",
       "Bootstrap: 2000 resamples, percentile 95% CI, seed 42. Chance = 0.25.",""]
    for title,key in SECTIONS:
        rows=by.get(key,[]); p,lo,hi,n=acc_ci([r["pred"] for r in rows],[r["gold"] for r in rows])
        L.append("## %s" % title); L.append("  accuracy = %.4f  (95%% CI %.4f-%.4f, n=%d)" % (p,lo,hi,n)); L.append("")
    print("\n".join(L).rstrip("\n"))
if __name__=="__main__": main()
