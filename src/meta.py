import csv, json
from pathlib import Path
OUT = Path("results/real"); GIT = "7e1fdee4db88c5b61792f424d0b27f33c35a87fc"
rows = list(csv.DictReader(open(OUT/"curve.csv")))

def sidecar(name, columns, ac):
    (OUT/(name+".md")).write_text(
"""# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (logits from src/extract_logits.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: %s
Seeds: 42 (MMLU eval sampling in src/task.py; greedy single forward, deterministic; 5-fold StratifiedKFold for the gold-fit transforms; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice (1000 items) via src/task.py; Qwen2.5-1.5B-Instruct base=checkpoint-0 and outcome-only GRPO RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; 4 option-letter logits at the 'Answer: (' position in results/real/logits_{base,rl}.npz
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, ac, columns))

sidecar("curve.csv",
        "  metric (base_acc/rl_acc/affine_to_rl_acc/affine_to_gold_acc/full_linear_gold_acc = MMLU accuracies 0-1; affine_to_rl_R2 = variance of RL option logits explained by the global affine map of base logits; affine_temperature_a = fitted scalar temperature; elicitation_fraction = (affine_to_gold_acc-base_acc)/(rl_acc-base_acc), ill-defined when the GRPO gain is within noise; rl_minus_base_acc = GRPO accuracy gain; identity_R2 = variance of RL logits explained by base logits with NO transform; bias_only_R2 = with a per-option constant bias and temperature fixed at 1; mean_abs_logit_delta = mean absolute base-to-RL option-logit change);\n"
        "  value; n (paired eval items, 1000)",
        "cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the 5 accuracies are reproduced from eval_points.jsonl, the R2/temperature/fraction rows are this file's data of record (figures 2-4)")
sidecar("eval_points.jsonl",
        "  section (one of base/rl/affine_to_rl/affine_to_gold/full_linear_gold); eval_order (position); pred (predicted option index 0-3); gold (correct option index 0-3)",
        "cd results/real && python3 recompute.py  (each section accuracy = mean(pred==gold) with 95% bootstrap CI)")

def w(stem, metrics, desc):
    rs=[r for r in rows if r["metric"] in metrics]
    with open(OUT/(stem+".csv"),"w",newline="") as f:
        wr=csv.DictWriter(f, fieldnames=["metric","value","n"]); wr.writeheader()
        for r in rs: wr.writerow(r)
    (OUT/(stem+".md")).write_text("# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/analyze.py + src/meta.py. Git-Commit: %s\n"%(stem,stem,desc,GIT))

w("figure_main", {"base_acc","rl_acc","affine_to_gold_acc","full_linear_gold_acc"}, "MMLU accuracy: base, RL (GRPO), global-affine-to-gold, full-linear ceiling.")
w("figure_affine_fit", {"affine_to_rl_R2","affine_temperature_a"}, "Global affine (temperature+per-option bias) fit of base option logits to RL option logits; R2 and temperature.")
w("figure_delta", {"mean_abs_logit_delta"}, "Per-option base-to-RL logit shift distribution; mean absolute delta.")
w("figure_variance", {"affine_to_rl_R2","identity_R2","bias_only_R2"}, "Variance of RL logits explained: identity (no transform), bias-only, full global affine.")

(OUT/"sources.json").write_text(json.dumps({"metrics":{"*":{"csv":"curve.csv"}},"per_example":["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")
