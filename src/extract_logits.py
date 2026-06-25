"""extract_logits.py — base vs RL 4-option answer logits on MMLU.

For base (checkpoint-0) and RL (checkpoint-final) we prefill the assistant turn
with 'Answer: (' and read the logits over the four option-letter tokens at that
position: the model's direct answer distribution. Saved per item with a shared
item_id so the analysis can pair the two models. No generation, one forward per
item, so this is fast and light. Reuses src/task.py. Python 3.10.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SUFFIX = "Answer: ("


def opt_ids(tok):
    return [tok.encode(L, add_special_tokens=False)[-1] for L in task.LABELS]


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter and adapter.lower() != "none":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter); m = m.merge_and_unload()
    m.eval()
    return m, tok


@torch.no_grad()
def run(label, adapter, items, out, batch=16):
    m, tok = load_model(adapter)
    oid = opt_ids(tok)
    prompts = [task.make_prompt_text(tok, it) + SUFFIX for it in items]
    logits = np.zeros((len(items), 4), dtype=np.float32)
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(m.device)
        out_l = m(**enc).logits[:, -1, :]  # left pad => last token is the prefill ')'-preceding pos
        logits[i:i+len(enc["input_ids"])] = out_l[:, oid].float().cpu().numpy()
    np.savez_compressed(out / ("logits_%s.npz" % label),
                        logits=logits,
                        gold=np.array([task.LABELS.index(it["gold_label"]) for it in items], dtype=np.int64),
                        item_id=np.arange(len(items), dtype=np.int64))
    acc = float((logits.argmax(1) == np.array([task.LABELS.index(it["gold_label"]) for it in items])).mean())
    print("[%s] n=%d acc=%.4f" % (label, len(items), acc), flush=True)
    del m; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_adapter", default="../rl-selfcot-causal/runs/rl/checkpoint-0")
    ap.add_argument("--rl_adapter", default="../rl-selfcot-causal/runs/rl/checkpoint-final")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/real")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    items = task.load_items("mmlu", "eval", n=args.n_eval, seed=args.seed)
    print("[eval] %d items" % len(items), flush=True)
    run("base", args.base_adapter, items, out)
    run("rl", args.rl_adapter, items, out)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
