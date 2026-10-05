"""E28 -- is the verdict directional? (mock-review W-C6, reviewer A Q2)

WHY. Lineage is directional: the question a BOM edge answers is which model is the
PARENT. Both reviewers ask whether the tool can express that at all. If
sigma_id(A,B) == sigma_id(B,A) then the score carries no direction, and the paper's
rank-based evasion criterion ("is the true parent in the top k") silently becomes
"is a relative in the top k".

This runs every pair in BOTH argument orders and reports the gap. It is a property
of the scoring function, so a handful of pairs across the relationship classes
settles it; no large corpus is needed.

Small models only (<= 1 GB each), so this arm needs ~4 GB, not the 16 GB the 6.9B
arms need. Run `preflight.py --need 5` first.
"""
import os, sys, json, time, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
OUT  = ROOT / "M9" / "e28_symmetry.jsonl"

# One pair per relationship class, all already in the local cache.
PAIRS = [
    ("derived",      "gpt2",                      "distilgpt2"),
    ("same-recipe",  "EleutherAI/pythia-70m",     "EleutherAI/pythia-70m-deduped"),
    ("same-recipe",  "google/multiberts-seed_0",  "google/multiberts-seed_1"),
    ("unrelated",    "gpt2",                      "bigscience/bloom-560m"),
    ("identity",     "gpt2",                      "gpt2"),
]

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}

def main():
    done = set()
    if OUT.exists():
        for l in OUT.open():
            if l.strip(): done.add(json.loads(l)["pair"])
    print(f"  {'class':13}{'pair':44}{'sig(A,B)':>10}{'sig(B,A)':>10}{'|diff|':>9}  verdicts")
    with OUT.open("a") as fh:
        for cls, a, b in PAIRS:
            key = f"{a}|{b}"
            if key in done:
                print(f"  [already recorded] {key}"); continue
            t0 = time.time()
            fwd, rev = mpk(a, b), mpk(b, a)
            if "error" in fwd or "error" in rev:
                print(f"  {cls:13}{key[:44]:44} ERROR"); continue
            sf, sr = fwd["scores"], rev["scores"]
            d = abs(sf["identity_score"] - sr["identity_score"])
            same = sf["provenance_decision"] == sr["provenance_decision"]
            rec = {"pair": key, "class": cls,
                   "forward": sf, "reverse": sr,
                   "identity_gap": d, "verdict_agrees": same,
                   "seconds": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  {cls:13}{(a.split('/')[-1]+' / '+b.split('/')[-1])[:44]:44}"
                  f"{sf['identity_score']:>10.4f}{sr['identity_score']:>10.4f}{d:>9.2e}  "
                  f"{'identical' if same else 'DIFFER'}")
    print("DONE-E28")

if __name__ == "__main__":
    main()
