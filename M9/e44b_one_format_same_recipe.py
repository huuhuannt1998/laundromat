"""E44b -- one-format identity scores for the same-recipe pairs that are not read in safetensors
order as served: the six MultiBERTs pairs of E5 and the 6.9B Pythia pair of E23.

Why. E35-E40 show identical weights score differently by tensor order. As served:
  * google/multiberts-seed_0..4 ship only pytorch_model.bin on `main`, so E5's six pairs
    (M7/e5_same_recipe.jsonl, identity 0.7633-0.7756) were read .bin against .bin;
  * EleutherAI/pythia-6.9b serves sharded safetensors on `main`, but pythia-6.9b-deduped serves
    only .bin shards, so E23's 6.9B row (identity 0.8211) was read safetensors against .bin.
The canonical fixed format for the one-format re-check is safetensors order on both sides (as
E36/E39/E40/E41/E44). Each .bin-only checkpoint is re-saved from its own main-revision .bin with
every tensor kept. For pythia-6.9b-deduped the shard layout and the sorted weight_map are taken
from the hub's own model.safetensors.index.json for that repo (cached, snapshot d7e0e808), so the
converted copy has the layout of the hub conversion; tensors are read with torch.load(mmap=True).

Reproduction controls first: one MultiBERTs pair and the 6.9B pair from the hub ids, and one
MultiBERTs pair from local .bin directories, to show the harness reproduces the frozen values and
that passing a local directory is neutral.

Invocation as E5/E23 (first model a, second b): pinned CLI, `provenancekit compare a b --json
--no-cache`, fully offline (HF_HUB_OFFLINE=1, uv --offline). Nothing is downloaded or sent.
Writes a NEW file, M9/e44b_one_format_same_recipe.jsonl. Copies in M9/work_e44b, deleted after.
"""
import os, json, shutil, pathlib, subprocess, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["UV_OFFLINE"] = "1"
import torch
from safetensors.torch import save_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e44b"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e44b_one_format_same_recipe.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
M = "google/multiberts-seed_"
MB_PAIRS = [(f"{M}0", f"{M}1"), (f"{M}0", f"{M}2"), (f"{M}1", f"{M}2"),
            (f"{M}2", f"{M}3"), (f"{M}3", f"{M}4"), (f"{M}0", f"{M}4")]
E5 = {(json.loads(l)["a"], json.loads(l)["b"]): json.loads(l)["identity"]
      for l in (ROOT/"M7"/"e5_same_recipe.jsonl").read_text().splitlines() if l.strip()}
P69, P69D = "EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped"
SKIP = (".safetensors", ".msgpack", ".h5", ".meta", ".index", ".pbtxt", ".ot")


def repo_dir(mid): return HUB / ("models--" + mid.replace("/", "--"))
def main_snapshot(mid):
    rev = (repo_dir(mid) / "refs" / "main").read_text().strip()
    return repo_dir(mid) / "snapshots" / rev, rev


def copy_meta(snap, d):
    for f in snap.iterdir():
        if f.name.endswith(SKIP) or f.name.startswith("pytorch_model") or "ckpt" in f.name \
           or f.name in ("README.md", "LICENSE", "model.safetensors.index.json"):
            continue
        shutil.copy(f.resolve(), d / f.name)


def materialise_single(mid, fmt):
    snap, rev = main_snapshot(mid)
    d = WORK / (mid.split("/")[-1] + "-" + fmt); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    copy_meta(snap, d); n = None
    if fmt == "bin":
        shutil.copy((snap / "pytorch_model.bin").resolve(), d / "pytorch_model.bin")
    else:
        sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(d / "model.safetensors"))
        n = len(sd); del sd
    return d, rev, n


def materialise_sharded_from_bin(mid):
    """Re-save a sharded .bin checkpoint as sharded safetensors using the hub's own index layout."""
    snap, rev = main_snapshot(mid)
    idx = next(p for p in (repo_dir(mid) / "snapshots").glob("*/model.safetensors.index.json"))
    hub_index = json.loads(idx.read_text())
    bin_index = json.loads((snap / "pytorch_model.bin.index.json").read_text())
    d = WORK / (mid.split("/")[-1] + "-safetensors-from-bin"); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    copy_meta(snap, d)
    bins = {f: torch.load(snap / f, map_location="cpu", weights_only=True, mmap=True)
            for f in sorted(set(bin_index["weight_map"].values()))}
    assert set(hub_index["weight_map"]) == set(bin_index["weight_map"]), "index tensor sets differ"
    for shard in sorted(set(hub_index["weight_map"].values())):
        names = sorted(k for k, v in hub_index["weight_map"].items() if v == shard)
        save_file({k: bins[bin_index["weight_map"][k]][k].contiguous() for k in names},
                  str(d / shard), metadata={"format": "pt"})
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": hub_index.get("metadata", {}), "weight_map": dict(sorted(hub_index["weight_map"].items()))},
        indent=2))
    del bins
    return d, rev, len(hub_index["weight_map"]), str(idx.parent.name)


def mpk(a, b):
    t0 = time.time()
    p = subprocess.run(["uv", "run", "--offline", "provenancekit", "compare", str(a), str(b), "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_OFFLINE": "1", "UV_OFFLINE": "1"})
    i = p.stdout.find("{")
    return ((json.loads(p.stdout[i:]), None) if i >= 0 else (None, (p.stderr or "")[-600:])), round(time.time()-t0, 1)


def emit(rec, res):
    (r, err), dt = res
    rec["seconds"] = dt
    if r is None: rec["error"] = err
    else: rec.update(scores=r["scores"], signals=r["signals"])
    with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
    print(f"{rec['a'].split('/')[-1]:<22}{rec['b'].split('/')[-1]:<22}{rec['arm']:<34} tier {s.get('mfi_tier')} "
          f"identity {s.get('identity_score')} nlf {g.get('nlf')} wvc {g.get('wvc')} -> {s.get('provenance_decision')} "
          f"({dt}s) {err[-200:] if err else ''}", flush=True)


OUT.write_text("")
# --- reproduction controls -------------------------------------------------------------------
a, b = MB_PAIRS[0]
emit(dict(kind="reproduction_control", suite="multiberts", a=a, b=b, arm="as-served (hub ids, both .bin)",
          expected_identity=E5[(a, b)], expected_source="M7/e5_same_recipe.jsonl"), mpk(a, b))
da, _, _ = materialise_single(a, "bin"); db, _, _ = materialise_single(b, "bin")
emit(dict(kind="reproduction_control", suite="multiberts", a=a, b=b, arm="both local .bin dirs",
          expected_identity=E5[(a, b)], expected_source="M7/e5_same_recipe.jsonl"), mpk(da, db))
shutil.rmtree(da, ignore_errors=True); shutil.rmtree(db, ignore_errors=True)
# --- MultiBERTs one-format --------------------------------------------------------------------
st = {}
for s in sorted({x for p in MB_PAIRS for x in p}):
    st[s] = materialise_single(s, "safetensors-from-bin")
for a, b in MB_PAIRS:
    emit(dict(kind="reread", suite="multiberts", a=a, b=b, arm="one-format (both st-from-bin)",
              a_main=st[a][1], b_main=st[b][1], n_tensors=st[a][2], as_served_identity=E5[(a, b)]),
         mpk(st[a][0], st[b][0]))
for s in st: shutil.rmtree(st[s][0], ignore_errors=True)
# --- Pythia 6.9B -------------------------------------------------------------------------------
emit(dict(kind="reproduction_control", suite="pythia", a=P69, b=P69D,
          arm="as-served (hub ids: a safetensors, b .bin)", expected_identity=0.8211,
          expected_source="M9/e23_scale.jsonl"), mpk(P69, P69D))
d, rev, n, idx_snap = materialise_sharded_from_bin(P69D)
emit(dict(kind="reread", suite="pythia", a=P69, b=P69D, arm="one-format (a hub st, b st-from-bin)",
          b_main=rev, n_tensors=n, shard_layout_from_snapshot=idx_snap, as_served_identity=0.8211),
     mpk(P69, d))
shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E44b")
