"""E7 -- systematic metadata sensitivity, weights held byte-identical throughout.

The missing-key result and the architecture-hash result currently read as two separate
bugs. This unifies them into one question: how much decision authority does each
publisher-controlled configuration field actually have? For one fixed parent/child pair
we vary ONLY config.json, one field at a time, and record what the verifier does.

Weights are never touched: every arm copies the same checkpoint directory and edits the
JSON. Absence, null and invalid values are tested separately, because they are different
publisher mistakes and the review asks for them to be distinguished.
"""
import os, sys, json, shutil, subprocess, pathlib, time, copy
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))


R    = pathlib.Path(_ROOT)
REPO = R/"M1"/"oracle"/"model-provenance-kit"
WORK = R/"M7"/"work_e7"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = R/"M7"/"e7_metadata_matrix.jsonl"

PARENT = "HuggingFaceTB/SmolLM2-135M"
CHILD  = "HuggingFaceTB/SmolLM2-135M-Instruct"

def mpk(args):
    p = subprocess.run(["uv", "run", "provenancekit", *args, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=5400,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    if i < 0:
        return {"__error__": (p.stderr or p.stdout or "")[-300:], "__exit__": p.returncode}
    return json.loads(p.stdout[i:])

def base_dir():
    d = WORK/"_base"
    if (d/"config.json").exists(): return d
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(CHILD); m.save_pretrained(d)
    AutoTokenizer.from_pretrained(CHILD).save_pretrained(d)
    del m
    return d

DELETE = object()
ARMS = [
    ("baseline",                    {}),
    ("architectures renamed",       {"architectures": ["LlamaForCausalLMExt"]}),
    ("architectures absent",        {"architectures": DELETE}),
    ("architectures null",          {"architectures": None}),
    ("architectures empty list",    {"architectures": []}),
    ("model_type renamed",          {"model_type": "llama_ext"}),
    ("model_type absent",           {"model_type": DELETE}),
    ("num_hidden_layers +1",        {"num_hidden_layers": 31}),
    ("hidden_size +64",             {"hidden_size": 640}),
    ("max_position_embeddings +1",  {"max_position_embeddings": 8193}),
    ("architectures + max_pos",     {"architectures": ["LlamaForCausalLMExt"],
                                     "max_position_embeddings": 8193}),
]

done = set()
if OUT.exists():
    done = {json.loads(l)["arm"] for l in OUT.read_text().splitlines() if l.strip()}

b = base_dir()
print(f"parent {PARENT}\nchild  {CHILD}  (weights identical in every arm)\n")
print(f"{'arm':<28}{'tier':>5}{'pipeline':>10}{'identity':>10}  verdict")
print("-"*82)
for name, edits in ARMS:
    if name in done:
        print(f"{name:<28}  (cached)"); continue
    d = WORK/name.replace(" ", "_").replace("+", "plus")
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(b, d)
    cfg = json.loads((d/"config.json").read_text())
    for k, v in edits.items():
        if v is DELETE: cfg.pop(k, None)
        else: cfg[k] = v
    (d/"config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.time()
    r = mpk(["compare", PARENT, str(d)])
    sc = r.get("scores", {}) if "__error__" not in r else {}
    rec = dict(arm=name, edits={k: ("<deleted>" if v is DELETE else v) for k, v in edits.items()},
               tier=sc.get("mfi_tier"), mfi_match=sc.get("mfi_match"),
               pipeline=sc.get("pipeline_score"), identity=sc.get("identity_score"),
               verdict=sc.get("provenance_decision"),
               error=(r.get("__error__") or "").replace(str(_ROOT), "<REPO>") or None,
               exit_code=r.get("__exit__"),
               secs=round(time.time()-t0, 1))
    with OUT.open("a") as fh: fh.write(json.dumps(rec)+"\n")
    shutil.rmtree(d, ignore_errors=True)
    if rec["error"]:
        print(f"{name:<28}{'--':>5}{'--':>10}{'--':>10}  NO VERDICT (exit {rec['exit_code']})")
    else:
        print(f"{name:<28}{str(rec['tier']):>5}{rec['pipeline']:>10}{rec['identity']:>10}  {rec['verdict']}")
print("\nDONE-E7")
