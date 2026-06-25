"""analyze_affine.py — can a single global affine logit transform of the base model
recover GRPO's effect / accuracy?

Arms (paired on item_id, all on the same MMLU eval set):
  base              : argmax of base option logits
  rl                : argmax of RL option logits
  affine_to_rl      : argmax of (a*base + b) where (a scalar, b in R^4) is the global
                      affine least-squares map base->RL logits (also reports R^2)
  affine_to_gold    : 5-fold OOF argmax after fitting (a,b) to minimize CE to gold
                      (the best a GLOBAL affine recalibration can do toward correctness)
  full_linear_gold  : 5-fold OOF multinomial logistic regression on the 4 base logits
                      (a full 4x4+bias map; the ceiling that a non-global transform reaches)
Reports the scalar elicitation fraction = (affine_to_gold_acc - base_acc)/(rl_acc - base_acc).
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED, CHANCE = 42, 0.25


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_global_affine_to_rl(Lb, Lr):
    # target Lr[i,c] ~ a*Lb[i,c] + b[c]; design = [Lb, onehot(c)] over all (i,c)
    n = Lb.shape[0]
    rows, tgt = [], []
    for c in range(4):
        oh = np.zeros((n, 4)); oh[:, c] = 1.0
        rows.append(np.column_stack([Lb[:, c], oh])); tgt.append(Lr[:, c])
    X = np.vstack(rows); y = np.concatenate(tgt)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = coef[0]; b = coef[1:5]
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Lhat = a * Lb + b[None, :]
    return a, b, float(r2), Lhat


def fit_global_affine_to_gold(Lb, gold, seed=SEED):
    # 5-fold OOF: params = [a, b0..b3], minimize CE(softmax(a*Lb+b), gold)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(gold), dtype=int)
    for tr, te in skf.split(Lb, gold):
        def nll(p):
            a, b = p[0], p[1:5]
            P = softmax(a * Lb[tr] + b[None, :])
            return -np.log(P[np.arange(len(tr)), gold[tr]] + 1e-12).mean()
        res = minimize(nll, np.array([1.0, 0, 0, 0, 0]), method="L-BFGS-B")
        a, b = res.x[0], res.x[1:5]
        oof[te] = (a * Lb[te] + b[None, :]).argmax(1)
    return oof


def fit_full_linear_gold(Lb, gold, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    from sklearn.model_selection import cross_val_predict
    return cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Lb, gold, cv=skf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default="results/real"); args = ap.parse_args()
    out = Path(args.out_dir)
    B = np.load(out / "logits_base.npz"); R = np.load(out / "logits_rl.npz")
    # pair on item_id
    bi = {int(x): k for k, x in enumerate(B["item_id"])}
    inter = sorted(set(B["item_id"].tolist()) & set(R["item_id"].tolist()))
    ri = {int(x): k for k, x in enumerate(R["item_id"])}
    Lb = np.array([B["logits"][bi[i]] for i in inter], float)
    Lr = np.array([R["logits"][ri[i]] for i in inter], float)
    gold = np.array([B["gold"][bi[i]] for i in inter], int)
    n = len(inter)
    print("[pair] n=%d" % n, flush=True)

    base_pred = Lb.argmax(1); rl_pred = Lr.argmax(1)
    a, b, r2, Lhat = fit_global_affine_to_rl(Lb, Lr)
    affrl_pred = Lhat.argmax(1)
    affgold_pred = fit_global_affine_to_gold(Lb, gold)
    fulllin_pred = fit_full_linear_gold(Lb, gold)

    arms = {"base": base_pred, "rl": rl_pred, "affine_to_rl": affrl_pred,
            "affine_to_gold": affgold_pred, "full_linear_gold": fulllin_pred}
    acc = {k: float((v == gold).mean()) for k, v in arms.items()}
    denom = acc["rl"] - acc["base"]
    elic = (acc["affine_to_gold"] - acc["base"]) / denom if abs(denom) > 1e-6 else float("nan")

    ep, rows = [], []
    for sec, pred in arms.items():
        for k in range(n):
            ep.append({"section": sec, "eval_order": k, "pred": int(pred[k]), "gold": int(gold[k])})
        rows.append((sec, acc[sec], n))
    import csv as _csv
    with open(out / "curve.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["metric", "value", "n"])
        for sec, ac, nn in rows:
            w.writerow(["%s_acc" % sec, "%.6f" % ac, nn])
        w.writerow(["affine_to_rl_R2", "%.6f" % r2, n])
        w.writerow(["affine_temperature_a", "%.6f" % a, n])
        w.writerow(["elicitation_fraction", ("%.6f" % elic) if elic == elic else "nan", n])
        w.writerow(["rl_minus_base_acc", "%.6f" % denom, n])
    with open(out / "eval_points.jsonl", "w") as f:
        for e in ep:
            f.write(json.dumps(e) + "\n")
    with open(out / "analysis_summary.txt", "w") as f:
        subprocess.run(["python3", "recompute.py"], cwd=str(out), stdout=f, check=True)
    make_figs(acc, r2, elic, a, b, Lb, Lr, n, out)
    print("[analyze] base=%.3f rl=%.3f affRL=%.3f(R2=%.3f) affGold=%.3f fullLin=%.3f elic=%.3f" % (
        acc["base"], acc["rl"], acc["affine_to_rl"], r2, acc["affine_to_gold"], acc["full_linear_gold"], elic), flush=True)


def make_figs(acc, r2, elic, a, b, Lb, Lr, n, out):
    # Fig1: accuracies
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ks = ["base", "rl", "affine_to_gold", "full_linear_gold"]
    labs = ["base", "RL (GRPO)", "global affine\n(to gold)", "full linear\n(ceiling)"]
    ax.bar(range(4), [acc[k] for k in ks], color=["#7570b3", "#d95f02", "#1b9e77", "#999999"])
    ax.axhline(CHANCE, color="k", ls=":", lw=1, label="chance")
    for i, k in enumerate(ks):
        ax.text(i, acc[k]+0.008, "%.3f" % acc[k], ha="center", fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylabel("MMLU accuracy"); ax.set_ylim(0, max(0.6, max(acc.values())+0.1))
    ax.set_title("Can a global affine transform of base logits match GRPO? (n=%d)" % n); ax.legend()
    fig.tight_layout(); fig.savefig(out/"figure_main.png", dpi=150); plt.close(fig)

    # Fig2: affine base->RL fit (scatter)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    Lhat = a*Lb + b[None,:]
    ax.scatter(Lr.ravel(), Lhat.ravel(), s=4, alpha=0.2, color="#d95f02")
    lo, hi = min(Lr.min(), Lhat.min()), max(Lr.max(), Lhat.max())
    ax.plot([lo,hi],[lo,hi], "k--", lw=1)
    ax.set_xlabel("RL option logit"); ax.set_ylabel("global affine of base"); ax.set_title("RL logits vs global affine of base (R2=%.3f)" % r2)
    fig.tight_layout(); fig.savefig(out/"figure_affine_fit.png", dpi=150); plt.close(fig)

    # Fig3: per-item RL-base logit delta distribution per option (is it a global shift?)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    delta = Lr - Lb
    ax.boxplot([delta[:, c] for c in range(4)], labels=["A","B","C","D"], showfliers=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("RL minus base option logit"); ax.set_title("Per-option logit shift from GRPO (a near-constant shift => global)")
    fig.tight_layout(); fig.savefig(out/"figure_delta.png", dpi=150); plt.close(fig)

    # Fig4: variance decomposition / elicitation summary
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.bar([0,1], [r2, 1-r2], 0.5, color=["#1b9e77","#e7298a"])
    ax.set_xticks([0,1]); ax.set_xticklabels(["explained by\nglobal affine", "residual\n(input-dependent)"])
    ax.set_ylabel("fraction of RL logit variance"); ax.set_ylim(0,1.0)
    ax.set_title("Is GRPO's logit change a global affine? R2=%.3f" % r2)
    fig.tight_layout(); fig.savefig(out/"figure_variance.png", dpi=150); plt.close(fig)
    print("[figures] wrote 4", flush=True)


if __name__ == "__main__":
    main()
