"""E47c -- the official AWM (github.com/LUMIA-Group/AWM, commit bc20ff8) on PolyPythias separate-seed
same-recipe pairs (review D1, F-01/F-03). Same harness as M7/awm/e3_awm_baseline.py (materialise via
AutoModelForCausalLM.save_pretrained; `main.py --model_paths A B --device cpu`; regexes unchanged),
offline from the HF cache. Reproduction control first: pythia-160m vs -deduped must give the frozen
E3 values (awm_wqwk 0.2438, z 85.72). Writes M7/awm/e47_awm_polypythias.jsonl (new file).
"""
import os, sys, json, re, subprocess, pathlib, gc, time
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
AWM = ROOT/"M7"/"awm"/"repo"; WORK = ROOT/"M7"/"awm"/"work"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"M7"/"awm"/"e47_awm_polypythias.jsonl"
PAT_AVG = re.compile(r"Average\s+(\S+)\s+Similarity\(%\)\s*=\s*([-\d.]+)")
PAT_Z = re.compile(r"Absolute Z-Score vs negative pairs\s*=\s*([\d.]+)")
P = "EleutherAI/pythia-"
E3 = {json.loads(l)["label"]: json.loads(l) for l in (ROOT/"M7/awm/e3_awm_results.jsonl").read_text().splitlines() if l.strip()}
PAIRS = [("pythia-160m vs -deduped (control)", P+"160m", P+"160m-deduped", "N-SR-shared-init")] + \
        [(f"pythia-{s} vs -seed1", f"{P}{s}", f"{P}{s}-seed1", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [(f"pythia-{s}-seed1 vs -seed2", f"{P}{s}-seed1", f"{P}{s}-seed2", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [(f"pythia-{s}-seed2 vs -seed3", f"{P}{s}-seed2", f"{P}{s}-seed3", "N-SR-seed") for s in ("70m", "160m", "410m")] + \
        [("pythia-160m vs data-seed1", P+"160m", P+"160m-data-seed1", "decoupled: data order only (same init)"),
         ("pythia-160m vs weight-seed1", P+"160m", P+"160m-weight-seed1", "decoupled: init only (same data order)")]

def materialise(rid):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = WORK/rid.replace("/", "--")
    if (d/"config.json").exists(): return d
    m = AutoModelForCausalLM.from_pretrained(rid); m.save_pretrained(d)
    try: AutoTokenizer.from_pretrained(rid).save_pretrained(d)
    except Exception: pass
    del m; gc.collect(); return d

def awm(p1, p2):
    r = subprocess.run([sys.executable, "main.py", "--model_paths", str(p1), str(p2), "--device", "cpu"],
                       cwd=AWM, capture_output=True, text=True, timeout=7200)
    avgs = {k: float(v)/100.0 for k, v in PAT_AVG.findall(r.stdout)}
    z = PAT_Z.search(r.stdout)
    return avgs, (float(z.group(1)) if z else None), r.stdout[-500:]

done = {json.loads(l)["label"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
for lab, ma, mb, cls in PAIRS:
    if lab in done: continue
    t0 = time.time(); pa, pb = materialise(ma), materialise(mb); avgs, z, tail = awm(pa, pb)
    rec = dict(label=lab, cls=cls, model_a=ma, model_b=mb, awm_wq=avgs.get("Wq_weights"), awm_wk=avgs.get("Wk_weights"),
               awm_wqwk=avgs.get("Wq_Wk_weights"), awm_z=z, status="OK" if avgs else "UNSUPPORTED",
               reason=None if avgs else tail[-250:], secs=round(time.time()-t0, 1))
    if "control" in lab:
        f = E3["pythia-160m vs -deduped"]; rec["expected"] = dict(awm_wqwk=f["awm_wqwk"], awm_z=f["awm_z"], source="M7/awm/e3_awm_results.jsonl")
        rec["reproduced"] = rec["awm_wqwk"] is not None and abs(rec["awm_wqwk"]-f["awm_wqwk"]) < 1e-9 and rec["awm_z"] == f["awm_z"]
    with OUT.open("a") as fh: fh.write(json.dumps(rec)+"\n")
    print(f"{cls:<10} {lab:<32} AWM={rec['awm_wqwk']} z={z} ({rec['secs']}s) {rec.get('reproduced','')}", flush=True)
print("DONE-E47c")
