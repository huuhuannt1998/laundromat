"""T5 — numerically verify that X1a, X1b, X2 are exact (function-preserving) at fp32."""
import os, sys, json, copy, torch, pathlib
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM, AutoTokenizer
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))


MODEL = sys.argv[1] if len(sys.argv)>1 else "HuggingFaceTB/SmolLM2-135M"
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL)
ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
print(f"model={MODEL}  hidden={ref.config.hidden_size}  layers={ref.config.num_hidden_layers} "
      f"heads={ref.config.num_attention_heads} kv={getattr(ref.config,'num_key_value_heads',None)} "
      f"inter={ref.config.intermediate_size} tied={ref.config.tie_word_embeddings}")

PROBES = [
 "The professor's arguments were criticized for their",
 "In 1969, the first humans landed on the surface of",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
 "The capital of France is Paris, and the capital of Germany is",
 "Water boils at 100 degrees Celsius at standard atmospheric",
 "She opened the letter carefully, unsure whether the news would be",
 "The mitochondria is often described as the powerhouse of the",
 "To be, or not to be, that is the",
] * 12   # 96 probe sequences

CASES = [
 ("X1a(frac=1.0)",   lambda m: LD.X1a_mlp_permute(m, frac=1.0, seed=1)),
 ("X1a(frac=0.5)",   lambda m: LD.X1a_mlp_permute(m, frac=0.5, seed=2)),
 ("X1b(heads)",      lambda m: LD.X1b_head_permute(m, seed=3)),
 ("X2(sigma=0.1)",   lambda m: LD.X2_norm_scale(m, log_sigma=0.1, seed=4)),
 ("X2(sigma=0.5)",   lambda m: LD.X2_norm_scale(m, log_sigma=0.5, seed=5)),
 ("X1a o X1b o X2",  lambda m: [LD.X1a_mlp_permute(m,1.0,6), LD.X1b_head_permute(m,6), LD.X2_norm_scale(m,0.1,6)]),
]
out=[]
for name, fn in CASES:
    m = copy.deepcopy(ref)
    meta = fn(m)
    r = LD.verify_exact(ref, m, tok, PROBES)
    ok = "EXACT  " if r["exact_within_tolerance"] else "*** NOT EXACT ***"
    print(f"{ok} {name:<18} max|dlogit|={r['max_abs_logit_delta']:.3e}  "
          f"ppl {r['ppl_ref']:.6f} -> {r['ppl_new']:.6f}  rel={r['rel_ppl_delta']:.3e}")
    out.append({"case":name,"meta":meta,"verify":r})
    del m
pathlib.Path(str(_ROOT) + "/M2/results/exactness_%s.json"
             % MODEL.replace("/","--")).write_text(json.dumps(out,indent=1,default=str))
print("DONE")
