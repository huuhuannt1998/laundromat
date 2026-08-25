"""E12 -- the training-progress ladder.

Section VII says we cannot express the displacement claim in units of training
compute because we do not know the mapping from displacement to compute.  Pythia
removes that excuse: EleutherAI released intermediate checkpoints of a single
training run, so checkpoint_k IS a descendant of checkpoint_j<k by construction,
and the compute separating them is known exactly (2,097,152 tokens/step:
1024 sequences x 2048 tokens).

For each checkpoint we ask the shipped verifier to compare it against the FINAL
checkpoint of the same run -- a lineage relationship no one disputes -- and record
what it says.  The dependent variable is the identity score, because config.json
is byte-identical across revisions and therefore pins the MFI tier for every pair;
that is the same reason tab:pricing uses the identity score.
"""
import os, sys, json, gc, shutil, pathlib, subprocess, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

REPO = str(_ROOT) + "/M1/oracle/model-provenance-kit"
WORK = pathlib.Path(str(_ROOT) + "/M6/work_e12"); WORK.mkdir(exist_ok=True)
OUT  = pathlib.Path(str(_ROOT) + "/M6/e12_pythia_ladder.jsonl")

MODEL      = "EleutherAI/pythia-160m"
FINAL_STEP = 143000
TOK_PER_STEP = 1024 * 2048          # 2,097,152
STEPS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000]

def mpk(args):
    p = subprocess.run(["uv", "run", "provenancekit", *args, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-400:]}

def materialise(step):
    """Save one revision to a local dir; returns the path."""
    d = WORK / f"step{step}"
    if d.exists(): return d
    rev = f"step{step}"
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev).eval()
    t = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    m.save_pretrained(d); t.save_pretrained(d)
    del m; gc.collect()
    return d

def displacement(a_dir, b_dir):
    """Relative RMS parameter displacement, overall and over embedding tensors.

    Matcher is 'embed' in the name, which selects embed_in/embed_out on GPT-NeoX.
    The roberta arm of tab:pricing matched 'embedding'; the tensor set is the same
    concept (token embedding matrices) but the name rule differs, so these values
    are reported on their own axis and are NOT pooled with that regression.
    """
    A = AutoModelForCausalLM.from_pretrained(a_dir).eval()
    B = AutoModelForCausalLM.from_pretrained(b_dir).eval()
    ad, bd = dict(A.named_parameters()), dict(B.named_parameters())
    num = den = enum = eden = 0.0
    for k, v in ad.items():
        if k not in bd or bd[k].shape != v.shape: continue
        d = (bd[k].float() - v.float()).pow(2).sum().item()
        n = v.float().pow(2).sum().item()
        num += d; den += n
        if "embed" in k:
            enum += d; eden += n
    del A, B; gc.collect()
    return {"overall": (num/den)**0.5 if den else float("nan"),
            "embedding": (enum/eden)**0.5 if eden else float("nan")}

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip(): done.add(json.loads(line)["step"])

print(f"=== E12 pythia ladder: {MODEL}, reference = step{FINAL_STEP} ===", flush=True)
final_dir = materialise(FINAL_STEP)
print(f"final checkpoint materialised at {final_dir}", flush=True)

for step in STEPS:
    if step in done:
        print(f"step{step}: already recorded, skipping", flush=True); continue
    t0 = time.time()
    try:
        d = materialise(step)
    except Exception as e:
        print(f"step{step}: DOWNLOAD FAILED {type(e).__name__}: {e}", flush=True); continue
    disp = displacement(d, final_dir)
    r = mpk(["compare", str(d), str(final_dir)])
    sc, sg = r.get("scores", {}), r.get("signals", {})
    rec = dict(step=step, final_step=FINAL_STEP, model=MODEL,
               tokens_at_step=step * TOK_PER_STEP,
               tokens_between=(FINAL_STEP - step) * TOK_PER_STEP,
               frac_of_run=step / FINAL_STEP,
               identity_score=sc.get("identity_score"),
               pipeline_score=sc.get("pipeline_score"),
               verdict=r.get("verdict") or r.get("classification"),
               tier=(r.get("mfi") or {}).get("tier") if isinstance(r.get("mfi"), dict) else r.get("tier"),
               signals={k: sg.get(k) for k in ("eas","wvc","end","lep","nlf")},
               d_overall=disp["overall"], d_embedding=disp["embedding"],
               error=r.get("error"), secs=round(time.time()-t0,1))
    with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
    sid = rec["identity_score"]
    print(f"step{step:<7} sig_id={('%.4f'%sid) if sid is not None else 'None':>7} "
          f"d_emb={disp['embedding']:.4f} d_all={disp['overall']:.4f} "
          f"verdict={rec['verdict']} ({rec['secs']}s)", flush=True)
    shutil.rmtree(d, ignore_errors=True)   # keep disk bounded
    gc.collect()

print("DONE-E12", flush=True)
