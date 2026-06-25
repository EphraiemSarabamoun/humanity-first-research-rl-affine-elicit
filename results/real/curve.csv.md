# curve.csv — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (logits from src/extract_logits.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: 7e1fdee4db88c5b61792f424d0b27f33c35a87fc
Seeds: 42 (MMLU eval sampling in src/task.py; greedy single forward, deterministic; 5-fold StratifiedKFold for the gold-fit transforms; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice (1000 items) via src/task.py; Qwen2.5-1.5B-Instruct base=checkpoint-0 and outcome-only GRPO RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; 4 option-letter logits at the 'Answer: (' position in results/real/logits_{base,rl}.npz
Analysis-Command: cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the 5 accuracies are reproduced from eval_points.jsonl, the R2/temperature/fraction rows are this file's data of record (figures 2-4)
Columns:
  metric (base_acc/rl_acc/affine_to_rl_acc/affine_to_gold_acc/full_linear_gold_acc = MMLU accuracies 0-1; affine_to_rl_R2 = variance of RL option logits explained by the global affine map of base logits; affine_temperature_a = fitted scalar temperature; elicitation_fraction = (affine_to_gold_acc-base_acc)/(rl_acc-base_acc), ill-defined when the GRPO gain is within noise; rl_minus_base_acc = GRPO accuracy gain; identity_R2 = variance of RL logits explained by base logits with NO transform; bias_only_R2 = with a per-option constant bias and temperature fixed at 1; mean_abs_logit_delta = mean absolute base-to-RL option-logit change);
  value; n (paired eval items, 1000)
