"""E47a -- initialization screen for same-recipe pairs (review D1, finding F-01).

Question: do the Pythia standard/deduplicated pairs share their initialization, and do the PolyPythias
separate-seed pairs not? Two readings per pair, both offline from the HF cache (fetched by e47_fetch.py):
  (1) step0 identity: every tensor of the two step0 checkpoints compared with torch.equal (all keys),
      plus sha256 of each step0 weight file;
  (2) the paper's weight screen, unchanged in method from M9/verify_label.py: mean over layers of the
      whole-matrix cosine of `mlp.dense_h_to_4h.weight`, models loaded with AutoModelForCausalLM in
      float16. Read at step0 and at the final (main) checkpoint.
Reproduction control: the final-checkpoint screen on pythia-{70m,160m,410m} vs -deduped must reproduce
M9/e37_pythia_cosine.log (0.1341, 0.1606, 0.1996).
Writes M9/e47/e47_init_screen.json (new file). Nothing is downloaded or sent.
"""
import os, json, hashlib, pathlib, statistics, gc, warnings, logging
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from safetensors.torch import load_file
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
KEY = "mlp.dense_h_to_4h.weight"
HERE = pathlib.Path(__file__).resolve().parent
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
P = "EleutherAI/pythia-"

def snap(mid, rev):
    d = HUB/("models--"+mid.replace("/", "--")); r = (d/"refs"/rev).read_text().strip()
    return d/"snapshots"/r, r

def wfile(s):
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (s/n).exists(): return s/n
    raise FileNotFoundError(s)

def raw_sd(mid, rev):
    f = wfile(snap(mid, rev)[0])
    return (load_file(str(f)) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu", weights_only=True)), f

def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

_mats = {}
def mats(mid, rev):                       # verify_label.py's extraction, with a revision
    k = (mid, rev)
    if k not in _mats:
        m = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16, low_cpu_mem_usage=True).eval()
        _mats[k] = np.stack([p.detach().numpy() for n, p in m.named_parameters() if KEY in n]); del m; gc.collect()
    return _mats[k]

def screen(a, ra, b, rb):                 # verify_label.py's statistic
    A, B = mats(a, ra), mats(b, rb); cs = []
    for i in range(min(len(A), len(B))):
        x = np.asarray(A[i], np.float32).ravel(); y = np.asarray(B[i], np.float32).ravel()
        cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return dict(layers=len(cs), mean_cos=round(statistics.mean(cs), 6), min_cos=round(min(cs), 6), max_cos=round(max(cs), 6))

def step0_identity(a, b):
    sa, fa = raw_sd(a, "step0"); sb, fb = raw_sd(b, "step0")
    ka = {k for k in sa if not k.endswith(("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq"))}
    kb = {k for k in sb if not k.endswith(("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq"))}
    common = sorted(ka & kb)
    eq = sum(bool(torch.equal(sa[k].float(), sb[k].float())) for k in common)
    maxdiff = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in common)
    out = dict(file_a=f"{fa.parent.name[:12]}/{fa.name}", file_b=f"{fb.parent.name[:12]}/{fb.name}",
               sha256_a=sha256(fa), sha256_b=sha256(fb), n_common=len(common), only_a=len(ka-kb), only_b=len(kb-ka),
               n_equal=eq, all_equal=(eq == len(common) and not (ka ^ kb)), max_abs_diff=maxdiff)
    del sa, sb; gc.collect(); return out

PAIRS = []
for s in ("70m", "160m", "410m"):
    PAIRS += [(f"{P}{s}", f"{P}{s}-deduped", "shared-init? (standard vs deduplicated)"),
              (f"{P}{s}", f"{P}{s}-seed1", "separate seed (PolyPythias)"),
              (f"{P}{s}-seed1", f"{P}{s}-seed2", "separate seed (PolyPythias)"),
              (f"{P}{s}-seed2", f"{P}{s}-seed3", "separate seed (PolyPythias)")]
PAIRS += [(f"{P}160m", f"{P}160m-data-seed1", "decoupled: data order varied, weight init fixed (publisher)"),
          (f"{P}160m", f"{P}160m-weight-seed1", "decoupled: weight init varied, data order fixed (publisher)")]
E37 = {"70m": 0.1341, "160m": 0.1606, "410m": 0.1996}

rows = []
for a, b, cls in PAIRS:
    r = dict(a=a, b=b, cls=cls, step0=step0_identity(a, b),
             screen_step0=screen(a, "step0", b, "step0"), screen_final=screen(a, "main", b, "main"))
    size = a.split("-")[1]
    if b.endswith("-deduped"):
        r["reproduction_control"] = dict(expected_final_mean_cos=E37[size], source="M9/e37_pythia_cosine.log",
                                         reproduced=round(r["screen_final"]["mean_cos"], 4) == E37[size])
    rows.append(r)
    print(f"{a.split('/')[-1]:>22} vs {b.split('/')[-1]:<24} step0 equal={r['step0']['all_equal']} "
          f"({r['step0']['n_equal']}/{r['step0']['n_common']}) cos0={r['screen_step0']['mean_cos']:.4f} "
          f"cosF={r['screen_final']['mean_cos']:.4f} {r.get('reproduction_control', '')}", flush=True)
    for k in [k for k in _mats if k[0] not in (a, b)]: _mats.pop(k)
(HERE/"e47_init_screen.json").write_text(json.dumps(dict(method="M9/verify_label.py statistic (mean whole-matrix cosine of mlp.dense_h_to_4h.weight, fp16 load) + torch.equal over all step0 tensors", rows=rows), indent=1))
print("DONE-E47a")
