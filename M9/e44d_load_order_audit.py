"""E44d -- audit: does each pair used in the one-format re-check (E45) load in the SAME tensor order
on both sides?

E44c showed the verifier's score depends on the order in which its loader returns tensors, and that
order depends on the load path, not only on the file format: a safetensors file read by load_file
and a .bin read through AutoModel can list the same tensors differently (and a sharded hub
safetensors repo fell back to AutoModel offline). E45 calls a pair "one-format" when both sides are
safetensors (re-saved child, or served as such). This checks that claim directly with the pinned
verifier's own loader (services.model_loader.load_state_dict), offline, for every parent and child
E45 uses: for the children that E45 takes from a re-saved .bin, the child is re-saved exactly as
E36/E39/E40/E44 did (save_file of the main-revision .bin, every tensor kept) and loaded back.

Per model it records the load source and the order of (a) norm tensors (what NLF concatenates) and
(b) tensors within each layer (what WVC concatenates), after stripping one leading wrapper name
(bert., roberta., model., gpt_neox., transformer., distilbert.). Per pair it reports whether the
relative order of the tensors BOTH sides hold agrees (task heads differ between parent and child, so
whole lists are not compared; a first version that compared whole lists was wrong and was discarded). Writes a NEW file, M9/e44d_load_order_audit.json. Re-saved
copies live in M9/work_e44d and are deleted after.
"""
import os, json, shutil, pathlib, gc, re, sys
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from safetensors.torch import save_file
from provenancekit.services.model_loader import load_state_dict
from provenancekit.config.settings import Settings
from provenancekit.utils.tensor import is_norm_tensor_name
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
WORK = ROOT / "M9" / "work_e44d"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e44d_load_order_audit.json"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
WRAP = ("bert.", "roberta.", "model.", "gpt_neox.", "transformer.", "distilbert.")
LAYER = re.compile(r"(?:^|\.)(?:layer|layers|h|block|blocks)\.(\d+)\.")

def strip(k):
    for w in WRAP:
        if k.startswith(w): return k[len(w):]
    return k

def orders(sd):
    keys = [strip(k) for k in sd.keys()]
    norm = [k for k, v in zip(keys, sd.values()) if is_norm_tensor_name(k) and v.dim() == 1 and v.numel() >= 64]
    per_layer = {}
    for k in keys:
        m = LAYER.search(k)
        if m: per_layer.setdefault((k.split(".layer")[0] if ".layer" in k else "", m.group(1)), []).append(k)
    return norm, {f"{a}|{b}": v for (a, b), v in per_layer.items()}

def main_snapshot(mid):
    d = HUB / ("models--" + mid.replace("/", "--"))
    rev = (d / "refs" / "main").read_text().strip(); return d / "snapshots" / rev

def resave(mid):
    snap = main_snapshot(mid); d = WORK / mid.replace("/", "--"); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for f in snap.iterdir():
        if f.name.endswith((".safetensors", ".msgpack", ".h5")) or f.name.startswith("pytorch_model") or f.name in ("README.md",):
            continue
        shutil.copy(f.resolve(), d / f.name)
    sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
    save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(d / "model.safetensors")); del sd
    return d

cache = {}
def probe(ref, label):
    if label in cache: return cache[label]
    try:
        r = load_state_dict(str(ref), settings=Settings())
        sd = r.state_dict
        rec = dict(ref=label, source=r.source, strategy=str(r.strategy))
        if sd is not None:
            n, pl = orders(sd); rec.update(n_norm=len(n), first_norm=n[:4]); cache[label] = (rec, n, pl)
        else:
            cache[label] = (rec, None, None)
        del sd, r; gc.collect()
    except Exception as exc:  # noqa: BLE001
        cache[label] = (dict(ref=label, error=str(exc)[-300:]), None, None)
    return cache[label]

# pairs: (parent, child, child_mode) where child_mode is "resaved" or "served"
PAIRS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
res = []
for parent, child, mode in PAIRS:
    if mode == "both_resaved":
        dp = resave(parent); prec, pn, pl = probe(dp, parent + " [re-saved]"); shutil.rmtree(dp, ignore_errors=True)
    else:
        prec, pn, pl = probe(parent, parent)
    if mode in ("resaved", "both_resaved"):
        d = resave(child); crec, cn, cl = probe(d, child + " [re-saved]"); shutil.rmtree(d, ignore_errors=True)
    else:
        crec, cn, cl = probe(child, child)
    # compare the relative order of the keys BOTH sides have (heads differ between parent and child)
    if pn is not None and cn is not None:
        shared = set(pn) & set(cn)
        same_norm = [k for k in pn if k in shared] == [k for k in cn if k in shared]
        n_shared_norm = len(shared)
    else:
        same_norm, n_shared_norm = None, 0
    common = sorted(set(pl or {}) & set(cl or {})) if (pl and cl) else []
    def rel(a, b):
        sh = set(a) & set(b); return [k for k in a if k in sh] == [k for k in b if k in sh]
    same_layer = all(rel(pl[k], cl[k]) for k in common) if common else None
    rec = dict(parent=parent, child=child, child_mode=mode, parent_source=prec.get("source"), child_source=crec.get("source"),
               same_norm_order=same_norm, n_shared_norm=n_shared_norm, same_within_layer_order=same_layer, n_common_layers=len(common),
               parent_first_norm=prec.get("first_norm"), child_first_norm=crec.get("first_norm"),
               errors=[x.get("error") for x in (prec, crec) if x.get("error")])
    res.append(rec); print(json.dumps(rec), flush=True)
OUT.write_text(json.dumps(dict(pairs=res, models=[c[0] for c in cache.values()]), indent=1))
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E44d")
