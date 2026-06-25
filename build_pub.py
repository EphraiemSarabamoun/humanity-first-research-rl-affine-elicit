import json
from pathlib import Path
R = Path("results/real")
gate = json.load(open(R/"gate_report.json")); panel = json.load(open(R/"reviewer_panel.json")); cost = json.load(open(R/"cost_report.json"))
p = {
 "plain": "There is a hope that training an AI only on whether its final answer is correct (not its reasoning) mostly just re-surfaces what the base model already knew, rather than teaching something new. We tested a sharp version: can a single simple rescaling of the base model answer scores reproduce what the RL training did? On a small model with a short RL run, RL barely changed the model at all: it flipped the predicted answer on under 3 percent of questions, with no real accuracy gain and no significant direction, and the tiny change it did make is almost perfectly a uniform rescaling (mostly a nudge toward option A). So there was essentially no gain to recover, and what little changed was a global recalibration, not a new capability. The catch is that the RL run was weak, so this does not settle the question for strong RL. Approve publishes it as an honest negative and methods result; Reject shelves it.",
 "cost": {"api_equivalent_usd": cost.get("api_equivalent_usd"), "sessions": cost.get("sessions")},
 "paper_title": "Small-Scale Outcome-Only GRPO Barely Moves the Answer Distribution: A Global-Affine Decomposition and a Paired Null",
 "headline_claim": "On Qwen2.5-1.5B-Instruct, 120 steps of outcome-only GRPO changed the predicted MMLU answer on only 27/1000 items (2.7%), with no significant direction (McNemar p=0.50) and no accuracy gain (0.378 to 0.382), so the scalar elicitation fraction is ill-defined. The small logit change it did make is near-pure global recalibration: a single affine transform explains 99.3% of the base-to-RL logit variance versus 98.5% with no transform, dominated by a per-option bias toward option A. A bounded null at a weak treatment, not a refutation; the affine-vs-identity-vs-full-linear decomposition is offered as a cheap diagnostic for stronger RL.",
 "weakest_part": "This is a bounded, weak-treatment null and I do not dress it up. (1) The dominant fact is that the RL barely changed the model (prediction flips on 2.7% of items, McNemar p=0.50), so the strong global-recalibration reading is partly the trivial fact that the model barely moved (identity already explains 98.5% of the logit variance; the affine adds about one point). I downgraded the title from recalibration-not-capability-change to barely-moves-the-answer-distribution for exactly this reason. (2) The supervised affine-to-gold and full-linear ceilings (0.389, 0.404) are fit on the eval labels in cross-validation, so they are NOT a fair head-to-head with unsupervised GRPO; I report them only as recalibration ceilings, not as beating GRPO. (3) The R-squared and per-option-shift numbers are point estimates without CIs (bootstrapping them needs per-resample refits); the paired McNemar test is the uncertainty-aware backbone of the no-change claim. (4) Weak treatment means low power: 120 GRPO steps on a 1.5B model; a regime where outcome-only RL transforms reasoning is the test that matters. (5) Four-option MMLU logits at one forced-answer position, not generative reasoning; one model, size, and reward. The reviewer panel lean-rejected the original title and the supervised-vs-unsupervised comparison; I added the paired test and reframed accordingly.",
 "gate_report": gate, "reviewer_panel": panel,
 "not_checked": [
   "a stronger RL treatment that produces a real accuracy gain (the regime where the elicitation fraction is defined)",
   "bootstrap CIs on the R-squared and per-option logit-shift estimates",
   "generative (non-multiple-choice) reasoning rather than forced-answer option logits",
   "larger models and other reward types (math/code verifiable rewards)",
   "an unsupervised base-to-RL ceiling fit to separate global from barely-changed more sharply",
   "whether the option-A bias reflects a position/label prior versus content"]
}
open(R/"pub.json", "w").write(json.dumps(p, indent=1))
print("pub.json bytes", len(json.dumps(p)), "cost", p["cost"]["api_equivalent_usd"])
