# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (logits from src/extract_logits.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: 7e1fdee4db88c5b61792f424d0b27f33c35a87fc
Seeds: 42 (MMLU eval sampling in src/task.py; greedy single forward, deterministic; 5-fold StratifiedKFold for the gold-fit transforms; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice (1000 items) via src/task.py; Qwen2.5-1.5B-Instruct base=checkpoint-0 and outcome-only GRPO RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; 4 option-letter logits at the 'Answer: (' position in results/real/logits_{base,rl}.npz
Analysis-Command: cd results/real && python3 recompute.py  (each section accuracy = mean(pred==gold) with 95% bootstrap CI)
Columns:
  section (one of base/rl/affine_to_rl/affine_to_gold/full_linear_gold); eval_order (position); pred (predicted option index 0-3); gold (correct option index 0-3)
