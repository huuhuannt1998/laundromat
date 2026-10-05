"""E44c -- why does the 6.9B pair read NLF 0.9976 as served but 0.0059 after re-saving
pythia-6.9b-deduped as safetensors (E44b)? A diagnostic, not a new score.

For three references -- the hub id EleutherAI/pythia-6.9b (main serves sharded safetensors), the hub
id EleutherAI/pythia-6.9b-deduped (main serves sharded .bin), and a local re-save of the deduped
.bin as sharded safetensors with the hub's own index layout (as E44b) -- this calls the pinned
verifier's own loader (services.model_loader.load_state_dict) and its own NLF extractor
(core.signals.weight_signals._extract_nlf), and records: load strategy and source, number of
state_dict keys, the first norm-tensor names in iteration order, the NLF mode and length. It then
computes the verifier's NLF similarity for each pair of the three. Run inside the verifier's venv,
offline. Writes a NEW file, M9/e44c_6p9b_diagnostic.json. The re-saved copy is deleted after.
"""
import os, json, shutil, pathlib, gc
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch
from safetensors.torch import save_file
from provenancekit.services.model_loader import load_state_dict
from provenancekit.core.signals.weight_signals import _extract_nlf, nlf_similarity
from provenancekit.config.settings import Settings
from provenancekit.models.signals import WeightSignalFeatures  # noqa: F401  (type of features)
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
WORK = ROOT / "M9" / "work_e44c"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e44c_6p9b_diagnostic.json"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
P69, P69D = "EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped"
from provenancekit.core.signals.weight_signals import is_norm_tensor_name  # noqa: E402


def resave_deduped():
    rd = HUB / ("models--" + P69D.replace("/", "--"))
    rev = (rd / "refs" / "main").read_text().strip(); snap = rd / "snapshots" / rev
    hub_index = json.loads(next((rd / "snapshots").glob("*/model.safetensors.index.json")).read_text())
    bin_index = json.loads((snap / "pytorch_model.bin.index.json").read_text())
    d = WORK / "pythia-6.9b-deduped-safetensors-from-bin"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for f in snap.iterdir():
        if f.name.startswith("pytorch_model") or f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            continue
        shutil.copy(f.resolve(), d / f.name)
    bins = {f: torch.load(snap / f, map_location="cpu", weights_only=True, mmap=True)
            for f in sorted(set(bin_index["weight_map"].values()))}
    for shard in sorted(set(hub_index["weight_map"].values())):
        names = sorted(k for k, v in hub_index["weight_map"].items() if v == shard)
        save_file({k: bins[bin_index["weight_map"][k]][k].contiguous() for k in names}, str(d / shard),
                  metadata={"format": "pt"})
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": hub_index.get("metadata", {}), "weight_map": dict(sorted(hub_index["weight_map"].items()))}, indent=2))
    del bins; gc.collect()
    return d


class F:  # minimal stand-in carrying the three NLF fields the verifier's similarity reads
    def __init__(self, v, mode, n): self.nlf_vector, self.nlf_mode, self.nlf_num_layers = v, mode, n


def probe(ref):
    r = load_state_dict(str(ref), settings=Settings())
    rec = dict(ref=str(ref).replace(str(WORK), "M9/work_e44c"), strategy=str(r.strategy), source=r.source)
    sd = r.state_dict
    if sd is None:
        rec["note"] = "no state_dict (streaming)"; return rec, None
    keys = list(sd.keys())
    norm_keys = [k for k in keys if is_norm_tensor_name(k) and sd[k].dim() == 1 and sd[k].numel() >= 64]
    v, mode, n = _extract_nlf(sd)
    rec.update(n_keys=len(keys), first_keys=keys[:4], n_norm=len(norm_keys), first_norm_keys=norm_keys[:6],
               norm_keys_sorted=norm_keys == sorted(norm_keys), nlf_mode=mode, nlf_len=int(len(v)) if v is not None else None,
               nlf_mean=float(np.mean(v)) if v is not None else None)
    feat = F(v, mode, n)
    del sd, r; gc.collect()
    return rec, (feat, norm_keys)


d = resave_deduped()
res, feats = [], {}
for name, ref in (("hub_6.9b", P69), ("hub_6.9b-deduped", P69D), ("resaved_6.9b-deduped", d)):
    rec, f = probe(ref); rec["name"] = name; res.append(rec); feats[name] = f
    print(json.dumps(rec), flush=True)
pairs = {}
for a, b in (("hub_6.9b", "hub_6.9b-deduped"), ("hub_6.9b", "resaved_6.9b-deduped"), ("hub_6.9b-deduped", "resaved_6.9b-deduped")):
    if feats[a] and feats[b]:
        pairs[f"{a} | {b}"] = dict(nlf_similarity=nlf_similarity(feats[a][0], feats[b][0]),
                                   same_norm_key_order=feats[a][1] == feats[b][1])
print(json.dumps(pairs, indent=1))
OUT.write_text(json.dumps(dict(models=res, pairs=pairs), indent=1))
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E44c")
