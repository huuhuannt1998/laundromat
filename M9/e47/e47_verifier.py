"""E47b -- the pinned verifier on PolyPythias separate-seed same-recipe pairs (review D1, F-01).

Pythia's standard and deduplicated runs share their step-0 checkpoint (E47a), so they differ in data,
not in seed. PolyPythias (van der Wal et al., ICLR 2025; EleutherAI/pythia-{size}-seed{1-9}) retrains
Pythia at one size with the same recipe under different random seeds; its card documents the seed as
the only varied factor (and 160m-data-seedN / -weight-seedN vary only data order / only init).
Inclusion rule, fixed before running: same publisher-documented recipe and size, different seed,
step-0 checkpoints that differ (E47a). Sizes 70M, 160M, 410M (the PolyPythias sizes that fit here).

Readings, exactly as the paper's existing pairs:
  * as served: pinned CLI at a75007d, `provenancekit compare a b --json --no-cache`, hub ids, offline
    (as M1/results/pairs.jsonl and M7/e5_same_recipe.py);
  * one tensor order: both sides re-written by safetensors save_file, `compare child parent` (child = b),
    as M9/e44e_canonical_order.py (functions copied unchanged).
Reproduction controls first: pythia-160m vs pythia-160m-deduped in both arms must reproduce the frozen
values (M1/results/pairs.jsonl 0.6745; M9/e44e_canonical_order.jsonl).
Writes M9/e47/e47_verifier.jsonl (new file). Nothing is sent; nothing is downloaded (offline).
"""
import os, json, shutil, pathlib, subprocess, time, gc
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["UV_OFFLINE"] = "1"
import torch
from safetensors.torch import save_file, load_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work_e47b"; WORK.mkdir(parents=True, exist_ok=True)
OUT = HERE / "e47_verifier.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
P = "EleutherAI/pythia-"

def snap(mid):
    d = HUB / ("models--" + mid.replace("/", "--")); rev = (d/"refs"/"main").read_text().strip()
    return d/"snapshots"/rev, rev

def tload(f):
    try: return torch.load(f, map_location="cpu", weights_only=True, mmap=True)
    except Exception: return torch.load(f, map_location="cpu", weights_only=True)

def weights(s):
    names = {p.name for p in s.iterdir()}
    if "model.safetensors" in names: return load_file(str(s/"model.safetensors")), "model.safetensors"
    if "model.safetensors.index.json" in names:
        idx = json.loads((s/"model.safetensors.index.json").read_text())["weight_map"]; sd = {}
        for f in sorted(set(idx.values())): sd.update(load_file(str(s/f)))
        return sd, "sharded safetensors"
    if "pytorch_model.bin" in names: return tload(s/"pytorch_model.bin"), "pytorch_model.bin"
    raise FileNotFoundError(f"no weights in {s}")

made = {}
def canonical(mid):
    if mid in made: return made[mid]
    s, rev = snap(mid); d = WORK / mid.replace("/", "--"); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for f in s.iterdir():
        if f.name.endswith((".safetensors", ".msgpack", ".h5", ".ot")) or f.name.startswith("pytorch_model") \
           or f.name in ("model.safetensors.index.json", "README.md") or not f.is_file():
            continue
        shutil.copy(f.resolve(), d/f.name)
    sd, src = weights(s); out, seen = {}, set()
    for k, v in sd.items():
        ptr = v.untyped_storage().data_ptr()
        if ptr in seen: v = v.clone()
        seen.add(ptr); out[k] = v.contiguous()
    save_file(out, str(d/"model.safetensors")); n = len(out); del sd, out; gc.collect()
    made[mid] = (d, rev, src, n); return made[mid]

def mpk(a, b):
    t0 = time.time()
    p = subprocess.run(["uv", "run", "--offline", "provenancekit", "compare", str(a), str(b), "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_OFFLINE": "1", "UV_OFFLINE": "1"})
    i = p.stdout.find("{")
    return ((json.loads(p.stdout[i:]), None) if i >= 0 else (None, (p.stderr or "")[-600:])), round(time.time()-t0, 1)

FROZEN_AS = {}
for l in (ROOT/"M1/results/pairs.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l)
        if r.get("model_a") == f"{P}160m" and r.get("model_b") == f"{P}160m-deduped" and "result" in r:
            FROZEN_AS = r["result"]["scores"]; FROZEN_AS_SIG = r["result"]["signals"]
FROZEN_CAN = next(json.loads(l) for l in (ROOT/"M9/e44e_canonical_order.jsonl").read_text().splitlines()
                  if l.strip() and json.loads(l)["child"] == f"{P}160m-deduped")

CONTROL = [(f"{P}160m", f"{P}160m-deduped", "reproduction control (shared init, differs in data)")]
PAIRS = [(f"{P}{s}", f"{P}{s}-seed1", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [(f"{P}{s}-seed1", f"{P}{s}-seed2", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [(f"{P}{s}-seed2", f"{P}{s}-seed3", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [(f"{P}160m", f"{P}160m-data-seed1", "decoupled: data order only (same init)"),
         (f"{P}160m", f"{P}160m-weight-seed1", "decoupled: init only (same data order)")]

if not OUT.exists(): OUT.write_text("")
done = {(json.loads(l)["a"], json.loads(l)["b"], json.loads(l)["arm"]) for l in OUT.read_text().splitlines() if l.strip()}

def emit(rec, r, err):
    if r is None: rec["error"] = err
    else: rec.update(scores=r["scores"], signals=r["signals"])
    with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
    print(f"{rec['arm'][:9]:<9} {rec['a'].split('/')[-1]:>18} / {rec['b'].split('/')[-1]:<22} tier {s.get('mfi_tier')} "
          f"id {s.get('identity_score')} wvc {g.get('wvc')} nlf {g.get('nlf')} end {g.get('end')} -> {s.get('provenance_decision')} "
          f"({rec['seconds']}s) {(err or '')[-200:]}", flush=True)

for a, b, cls in CONTROL + PAIRS:
    if (a, b, "as-served") not in done:
        (r, err), dt = mpk(a, b)
        rec = dict(a=a, b=b, cls=cls, arm="as-served", invocation="compare a b (hub ids, offline)",
                   a_main=snap(a)[1], b_main=snap(b)[1], seconds=dt)
        if (a, b) == CONTROL[0][:2]:
            rec["expected"] = dict(identity_score=FROZEN_AS.get("identity_score"), source="M1/results/pairs.jsonl")
            rec["reproduced"] = bool(r) and r["scores"]["identity_score"] == FROZEN_AS.get("identity_score") and r["signals"] == FROZEN_AS_SIG
        emit(rec, r, err)
    if (a, b, "one-order") not in done:
        pd, prev, psrc, pn = canonical(a); cd, crev, csrc, cn = canonical(b)
        (r, err), dt = mpk(cd, pd)
        rec = dict(a=a, b=b, cls=cls, arm="one-order", invocation="compare b a (both re-written by save_file)",
                   a_main=prev, b_main=crev, a_weights_from=psrc, b_weights_from=csrc, n_tensors_a=pn, n_tensors_b=cn, seconds=dt)
        if (a, b) == CONTROL[0][:2]:
            rec["expected"] = dict(identity_score=FROZEN_CAN["scores"]["identity_score"], source="M9/e44e_canonical_order.jsonl")
            rec["reproduced"] = bool(r) and r["scores"] == FROZEN_CAN["scores"] and r["signals"] == FROZEN_CAN["signals"]
        emit(rec, r, err)
        for m in (a, b):
            shutil.rmtree(made.pop(m)[0], ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E47b")
