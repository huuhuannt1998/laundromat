"""E47 summary -- reads the E47 result files and writes M9/e47/e47_summary.json with every number the
revision cites and the sha256 of each source file. Read-only over the result files."""
import json, hashlib, pathlib, csv
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())
H = pathlib.Path(__file__).resolve().parent
FILES = {"verifier": "M9/e47/e47_verifier.jsonl", "init_screen": "M9/e47/e47_init_screen.json",
         "trajectory": "M9/e47/e47_decoupled_trajectory.json", "awm": "M7/awm/e47_awm_polypythias.jsonl",
         "tester_log": "M6/e47_tester_2000.log", "tester_csv": "M6/e47_tester_2000.csv"}
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
jl = lambda p: [json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
V = jl(FILES["verifier"]); A = jl(FILES["awm"]); S = json.loads((ROOT/FILES["init_screen"]).read_text())["rows"]
T = json.loads((ROOT/FILES["trajectory"]).read_text())["rows"]
seed = [r for r in V if r["cls"] == "N-SR-seed"]
def rng(xs): return [min(xs), max(xs)]
out = {"sha256": {k: sha(v) for k, v in FILES.items()}, "files": FILES}
for arm in ("as-served", "one-order"):
    rs = [r for r in seed if r["arm"] == arm]
    out[f"verifier_{arm}"] = dict(n=len(rs), tier1_confirmed=sum(r["scores"]["mfi_tier"] == 1 and r["scores"]["provenance_decision"] == "Confirmed Match" for r in rs),
        identity=rng([r["scores"]["identity_score"] for r in rs]), wvc=rng([r["signals"]["wvc"] for r in rs]),
        below_065=sorted(r["scores"]["identity_score"] for r in rs if r["scores"]["identity_score"] < 0.65),
        eas=rng([r["signals"]["eas"] for r in rs]), lep=rng([r["signals"]["lep"] for r in rs]),
        end=rng([r["signals"]["end"] for r in rs]), nlf=rng([r["signals"]["nlf"] for r in rs]),
        per_pair={f"{r['a'].split('/')[-1]}|{r['b'].split('/')[-1]}": r["scores"]["identity_score"] for r in rs})
out["verifier_controls"] = [dict(a=r["a"], b=r["b"], arm=r["arm"], reproduced=r.get("reproduced"), identity=r["scores"]["identity_score"],
                                 wvc=r["signals"]["wvc"], verdict=r["scores"]["provenance_decision"], tier=r["scores"]["mfi_tier"]) for r in V if r["cls"] != "N-SR-seed"]
aw = [r for r in A if r["cls"] == "N-SR-seed"]
out["awm_seed"] = dict(n=len(aw), raw=rng([r["awm_wqwk"] for r in aw]), z=rng([r["awm_z"] for r in aw]))
out["awm_controls"] = [dict(label=r["label"], raw=r["awm_wqwk"], z=r["awm_z"], reproduced=r.get("reproduced")) for r in A if r["cls"] != "N-SR-seed"]
out["init_screen"] = [dict(a=r["a"].split("/")[-1], b=r["b"].split("/")[-1], step0_all_equal=r["step0"]["all_equal"],
                           step0_equal=f"{r['step0']['n_equal']}/{r['step0']['n_common']}", step0_cos=r["screen_step0"]["mean_cos"],
                           final_cos=r["screen_final"]["mean_cos"], repro=(r.get("reproduction_control") or {}).get("reproduced")) for r in S]
sep = [r for r in S if "seed" in r["b"] and "data" not in r["b"] and "weight" not in r["b"]]
out["init_screen_seed"] = dict(step0_abs_cos_max=max(abs(r["screen_step0"]["mean_cos"]) for r in sep),
                               final_abs_cos_max=max(abs(r["screen_final"]["mean_cos"]) for r in sep),
                               any_step0_identical=any(r["step0"]["all_equal"] for r in sep))
out["trajectory"] = T
rows = list(csv.reader(open(ROOT/FILES["tester_csv"])))[1:]
tst = [dict(outcome=r[0].strip(), p=float(r[4]), hits=int(r[6]), n=int(r[7]), model=r[8].strip(), guess=r[10].strip()) for r in rows]
out["tester"] = tst
ts = [t for t in tst if "-seed" in t["model"] and "data" not in t["model"] and "weight" not in t["model"]]
out["tester_seed"] = dict(n=len(ts), named_parent=sum(t["outcome"] == "FP" for t in ts), p=[(t["model"].split("/")[-1], t["outcome"], t["p"]) for t in ts])
(H/"e47_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("trajectory", "tester", "init_screen")}, indent=1))
