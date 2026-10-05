"""E47e -- diagnostic for an unexpected E47a reading: pythia-160m-weight-seed1 (publisher: weight init varied,
data order fixed) has a step-0 checkpoint unrelated to pythia-160m's (screen -0.0002) yet reads 0.2111 at
the end of training. Read the same weight screen (M9/verify_label.py statistic) at intermediate checkpoints
to see when the position-wise similarity appears; data-seed1 (shared init, data order varied) and seed1
(both varied) are read alongside. Read-only downloads of the needed revisions; writes
M9/e47/e47_decoupled_trajectory.json (new file).
"""
import os, json, gc, pathlib, statistics, warnings, logging
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Two phases in two processes: `--fetch` only downloads (snapshot_download reads); the screen then runs in a
# fresh process with HF_HUB_OFFLINE=1 set before any HF import. An online from_pretrained of a .bin-only repo
# makes transformers ask the Hub's conversion service for a safetensors PR (first attempt,
# e47_decoupled_trajectory_aborted.log: the request carried no token and failed; the run was stopped).
import sys, subprocess
REVS = ("step1", "step64", "step1000", "step10000")
if "--fetch" in sys.argv:
    from huggingface_hub import snapshot_download, HfApi
    for _m in ("160m", "160m-weight-seed1", "160m-data-seed1", "160m-seed1"):
        for _r in REVS:
            _fs = [f.rfilename for f in HfApi().model_info("EleutherAI/pythia-" + _m, revision=_r).siblings]
            snapshot_download("EleutherAI/pythia-" + _m, revision=_r, max_workers=4,
                              allow_patterns=["config.json", "*.safetensors" if any(f.endswith(".safetensors") for f in _fs) else "*.bin"])
    print("FETCHED"); sys.exit(0)
if os.environ.get("HF_HUB_OFFLINE") != "1":
    subprocess.run([sys.executable, __file__, "--fetch"], check=True)
    os.execve(sys.executable, [sys.executable, __file__], {**os.environ, "HF_HUB_OFFLINE": "1"})
import numpy as np, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
KEY = "mlp.dense_h_to_4h.weight"; P = "EleutherAI/pythia-"
def mats(mid, rev):
    m = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    x = np.stack([p.detach().numpy() for n, p in m.named_parameters() if KEY in n]); del m; gc.collect(); return x
def screen(A, B):
    cs = [float(np.asarray(A[i], np.float32).ravel() @ np.asarray(B[i], np.float32).ravel() /
                (np.linalg.norm(np.asarray(A[i], np.float32)) * np.linalg.norm(np.asarray(B[i], np.float32)) + 1e-12)) for i in range(len(A))]
    return round(statistics.mean(cs), 6)
rows = []
for rev in ("step0",) + REVS + ("main",):
    ref = mats(P+"160m", rev)
    for other in (P+"160m-weight-seed1", P+"160m-data-seed1", P+"160m-seed1"):
        r = dict(revision=rev, a=P+"160m", b=other, mean_cos=screen(ref, mats(other, rev))); rows.append(r)
        print(rev, other.split("/")[-1], r["mean_cos"], flush=True)
pathlib.Path(__file__).with_name("e47_decoupled_trajectory.json").write_text(json.dumps(dict(method="M9/verify_label.py statistic", rows=rows), indent=1))
print("DONE-E47e")
