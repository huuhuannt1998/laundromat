"""Check a claimed parent->child relation in the WEIGHTS before trusting the label.

Every pair E23 uses must pass this. Two published labels have already failed it:
pythia-2.8b-deduped is a near-copy of pythia-2.8b (cos 0.997, so not an independent
run), and lambda/pythia-2.8b-deduped-synthetic-instruct sits at cos 0.19 from the
parent its card names (so not weight-derived). Running on an unverified label is how
the BERT-Tiny defect got in.

Memory-safe: one model resident at a time, up-projections spilled to disk in fp16.
"""
import sys, gc, pathlib, statistics, warnings, logging
import numpy as np, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
KEY = "mlp.dense_h_to_4h.weight"
TMP = pathlib.Path(__file__).parent / "reduced"; TMP.mkdir(exist_ok=True)

def spill(mid):
    dst = TMP / ("verify__" + mid.replace("/", "__") + ".npy")
    if dst.exists(): return dst
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16,
                                             low_cpu_mem_usage=True).eval()
    mats = [p.detach().numpy() for n, p in m.named_parameters() if KEY in n]
    del m; gc.collect()
    np.save(dst, np.stack(mats)); del mats; gc.collect()
    return dst

a, b = sys.argv[1], sys.argv[2]
A = np.load(spill(a), mmap_mode="r"); B = np.load(spill(b), mmap_mode="r")
cs = []
for i in range(min(len(A), len(B))):
    x = np.asarray(A[i], np.float32).ravel(); y = np.asarray(B[i], np.float32).ravel()
    cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
m = statistics.mean(cs)
print(f"  {a}\n  {b}")
print(f"  layers={len(cs)} mean cos={m:.4f} min={min(cs):.4f} max={max(cs):.4f}")
print("  VERDICT: " + ("weight-derived / near-copy" if m > 0.9 else
                       "NOT weight-related (independent-run range)"))
