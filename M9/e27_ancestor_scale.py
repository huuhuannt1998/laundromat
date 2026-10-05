"""E27 -- a genuine ancestor at 6.9B: the true-positive control E23 lacked.

WHY. E23 established the same-recipe FALSE POSITIVE at 6.9B (identity 0.8211,
Confirmed Match). It has no true positive at that scale, because all three published
derivative labels we screened failed weight verification. A reviewer can therefore ask
whether the tool simply says Confirmed Match to everything large.

Pythia's checkpoints answer it without trusting any published label: checkpoint_j is an
ancestor of the final model of the same run BY CONSTRUCTION. This is the instrument
Sec. VII already uses at 160m (E12).

WHAT IS AND IS NOT MEASURABLE HERE. config.json is byte-identical across revisions, so
the MFI gate fires at tier 1 and pins the pipeline score to 1.0 whatever the weights say
-- E12 found exactly that. The verdict is therefore uninformative and the dependent
variables are the IDENTITY SCORE and the LAP defence, as in E12 and tab:pricing.

The sharp question: does a genuine 6.9B ancestor score ABOVE the 0.8211 that the
same-recipe non-derivative earned? If not, the weight evidence ranks a false positive
above a true positive at this scale.

Materialised in float16, which is what MPK's loader pins anyway; E12 materialised in
float32 at 160m, so displacement values here are on their own axis and are not pooled
with E12's ladder.
"""
import os, sys, gc, json, time, shutil, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e27"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = ROOT / "M9" / "e27_ancestor_scale.jsonl"

MODEL = "EleutherAI/pythia-6.9b"
FINAL = "main"                 # the released final checkpoint
STEPS = [128000]               # ~90% of the run: a genuine ancestor with real distance
UP_KEYS = ("up_proj.weight", "dense_h_to_4h.weight", "c_fc.weight", "fc_in.weight")
MAX_FEAT = 1536                # identical to M4/p_tilde_arm.py and M9/e23_scale.py

def materialise(rev):
    d = WORK / (rev if rev != "main" else "final")
    if (d / "config.json").exists():
        return d
    # checkpoint revisions ship only pytorch_model-*.bin (no safetensors), so the
    # default safetensors path raises; fall back rather than fail the run.
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev,
                dtype=torch.float16, low_cpu_mem_usage=True).eval()
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev,
                dtype=torch.float16, low_cpu_mem_usage=True,
                use_safetensors=False).eval()
    t = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    m.save_pretrained(d); t.save_pretrained(d)
    del m; gc.collect()
    return d

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", str(a), str(b),
                        "--json", "--no-cache"], cwd=REPO, capture_output=True,
                       text=True, timeout=21600,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}

def reduced(path, tag):
    """Subsampled up-projections, SPILLED TO DISK and mmapped.

    Holding these in RAM (3.2 GB at 6.9B) while the MPK subprocess loads its own
    14 GB model gave a ~17 GB peak on a 24 GB laptop. Spilling keeps the peak at
    one model.
    """
    dst = WORK / f"reduced_{tag}.npy"
    if dst.exists():
        return np.load(dst, mmap_mode="r")
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                             low_cpu_mem_usage=True).eval()
    mats = []
    for n, p in m.named_parameters():
        if p.ndim == 2 and any(k in n for k in UP_KEYS):
            A = p.detach()
            if A.shape[1] > MAX_FEAT:
                f = torch.arange(0, A.shape[1], max(1, A.shape[1] // MAX_FEAT))[:MAX_FEAT]
                A = A[:, f]
            mats.append(A.clone().float().numpy())
    del m; gc.collect()
    np.save(dst, np.stack(mats)); del mats; gc.collect()
    return np.load(dst, mmap_mode="r")

def lap_colnorm(A, B):
    if A.shape != B.shape: return None
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())

def main():
    done = set()
    if OUT.exists():
        for l in OUT.open():
            if l.strip(): done.add(json.loads(l)["step"])
    print(f"=== E27: {MODEL}, ancestor checkpoints against the final ===", flush=True)
    fin = materialise(FINAL); print(f"  final materialised: {fin}", flush=True)

    with OUT.open("a") as fh:
        for step in STEPS:
            if step in done: print(f"  step{step} already recorded"); continue
            t0 = time.time()
            d = materialise(f"step{step}")
            print(f"  step{step} materialised", flush=True)
            # MPK first, with nothing of ours resident: its subprocess needs ~14 GB
            r = mpk(d, fin)
            if "error" in r:
                print(f"  step{step} MPK ERROR {r['error'][:200]}", flush=True); continue
            s, g = r["scores"], r["signals"]
            fin_red = reduced(fin, "final")
            red = reduced(d, f"step{step}")
            vals = [v for v in (lap_colnorm(red[i], fin_red[i])
                                for i in range(min(len(red), len(fin_red)))) if v is not None]
            lap = float(np.mean(vals)) if vals else float("nan")
            rec = {"step": step, "model": MODEL, "a": f"step{step}", "b": "final",
                   "scores": s, "signals": g, "lap_defence": lap,
                   "seconds": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  step{step}: tier={s['mfi_tier']} pipe={s['pipeline_score']:.4f} "
                  f"identity={s['identity_score']:.4f} -> {s['provenance_decision']}  "
                  f"LAP={lap:.4f}  ({rec['seconds']/60:.1f} min)", flush=True)
            del red; gc.collect()
    print("DONE-E27")

if __name__ == "__main__":
    main()
