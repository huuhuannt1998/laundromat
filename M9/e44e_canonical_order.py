"""E44e -- canonical-order re-read: every pair of the one-format re-check with BOTH sides re-serialised
by the same writer, so the verifier's loader returns tensors in the same (name) order on both sides.

Why. E44c/E44d showed that the verifier's NLF and WVC signals depend on the order in which its
loader returns tensors, and that this order is set by the file's writer, not by the file format:
  * a .bin is loaded through AutoModel and comes back in module order;
  * a safetensors file is read with load_file and comes back in its storage order, which is name
    order for files written by safetensors' save_file (bert-base-uncased, distilroberta-base, every
    re-saved copy) but not for others (roberta-base and gpt2 come back in neither name nor header
    order; pythia-70m in header order, which is not name order);
  * a sharded hub safetensors repo fell back to AutoModel when probed offline (pythia-1.4b-deduped,
    pythia-6.9b).
So "safetensors on both sides" (the E36/E39/E40/E41/E44 convention: re-save the child, read the
parent as served) does not guarantee one order: E44d finds the RoBERTa children, distilroberta-base,
all-distilroberta-v1, distilgpt2, pythia-1.4b and pythia-70m-v0 order-mismatched under it.

Canonical order, used here: load each model's own main-revision weights (any format, any sharding),
write them with safetensors.torch.save_file into one model.safetensors (every tensor kept; storage
shared between tensors is broken with clone), and compare the two re-written directories. Config and
tokenizer files are copied from the main snapshot; the 'architectures' repairs used by the
as-served runs are applied identically (Bio_ClinicalBERT -> BertForMaskedLM; AllenAI DAPT models ->
RobertaForMaskedLM). Pinned CLI, `compare child parent --json --no-cache`, offline. Writes a NEW
file, M9/e44e_canonical_order.jsonl. Re-written copies live in M9/work_e44e, deleted after.
"""
import os, json, shutil, pathlib, subprocess, time, gc, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1"); os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["UV_OFFLINE"] = "1"
import torch
from safetensors.torch import save_file, load_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e44e"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e44e_canonical_order.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
PATCH = {"emilyalsentzer/Bio_ClinicalBERT": {"architectures": ["BertForMaskedLM"]},
         **{f"allenai/{m}_roberta_base": {"architectures": ["RobertaForMaskedLM"]} for m in ("cs", "biomed", "news", "reviews")}}
PAIRS = json.loads(pathlib.Path(sys.argv[1]).read_text())   # [[parent, child], ...] in run order


def snap(mid):
    d = HUB / ("models--" + mid.replace("/", "--")); rev = (d/"refs"/"main").read_text().strip()
    return d/"snapshots"/rev, rev


def tload(f):
    try: return torch.load(f, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # legacy (non-zip) checkpoints cannot be memory-mapped
        return torch.load(f, map_location="cpu", weights_only=True)


def weights(s):
    names = {p.name for p in s.iterdir()}
    if "model.safetensors" in names: return load_file(str(s/"model.safetensors")), "model.safetensors"
    if "model.safetensors.index.json" in names:
        idx = json.loads((s/"model.safetensors.index.json").read_text())["weight_map"]; sd = {}
        for f in sorted(set(idx.values())): sd.update(load_file(str(s/f)))
        return sd, "sharded safetensors"
    if "pytorch_model.bin" in names:
        return tload(s/"pytorch_model.bin"), "pytorch_model.bin"
    if "pytorch_model.bin.index.json" in names:
        idx = json.loads((s/"pytorch_model.bin.index.json").read_text())["weight_map"]; sd = {}
        for f in sorted(set(idx.values())): sd.update(tload(s/f))
        return sd, "sharded .bin"
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
    if mid in PATCH:
        cfg = json.loads((d/"config.json").read_text()); cfg.update(PATCH[mid]); (d/"config.json").write_text(json.dumps(cfg, indent=1))
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


if not OUT.exists(): OUT.write_text("")
done = {(json.loads(l)["parent"], json.loads(l)["child"]) for l in OUT.read_text().splitlines() if l.strip()}
last_parent = None
for parent, child in PAIRS:
    if (parent, child) in done: continue
    if last_parent and last_parent != parent and last_parent in made:      # free disk for big parents
        shutil.rmtree(made.pop(last_parent)[0], ignore_errors=True)
    pd, prev, psrc, pn = canonical(parent); cd, crev, csrc, cn = canonical(child)
    (r, err), dt = mpk(cd, pd)
    rec = dict(parent=parent, child=child, arm="canonical (both re-written by save_file)", parent_main=prev, child_main=crev,
               parent_weights_from=psrc, child_weights_from=csrc, n_tensors_parent=pn, n_tensors_child=cn,
               config_patch_parent=PATCH.get(parent), config_patch_child=PATCH.get(child), seconds=dt)
    if r is None: rec["error"] = err
    else: rec.update(scores=r["scores"], signals=r["signals"])
    with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
    print(f"{child:<52} tier {s.get('mfi_tier')} identity {s.get('identity_score')} nlf {g.get('nlf')} wvc {g.get('wvc')} "
          f"lep {g.get('lep')} -> {s.get('provenance_decision')} ({dt}s) {err[-160:] if err else ''}", flush=True)
    shutil.rmtree(cd, ignore_errors=True); made.pop(child, None)
    last_parent = parent
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E44e")
