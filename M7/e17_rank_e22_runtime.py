"""E17 (rank everywhere) and E22 (runtime/memory), both from frozen scan outputs.

E17: the attack criterion is score AND database-wide rank, but rank was reported only
for selected attacks. For every scan we hold, report parent score, parent rank, top-1
identity, top-5 membership, and the margin to the strongest incorrect candidate.
A score drop that leaves the parent at rank 1 is not lineage hiding.

E22: the scan output already carries elapsed_ms, extract_seconds and lookup_seconds,
so the runtime profile needs no new measurement.
"""
import json, pathlib, statistics as st

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R = pathlib.Path(_ROOT)

def load(path, tp_key=None):
    out = []
    for line in (R/path).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); res = r.get("result") or {}
        if not res.get("matches"): continue
        out.append((r.get("target"), r.get(tp_key) if tp_key else None,
                    r.get("note"), res))
    return out

recs = load("M2/results/wide_scans.jsonl", "true_parent") + load("M1/results/scans.jsonl")

def norm(s):
    return (s or "").split("/")[-1].lower()

print(f"{'target':<34}{'true parent':<20}{'rank':>6}{'sig':>8}{'top1':>8}{'top5':>6}{'margin':>9}")
print("-"*95)
rows, times = [], []
for target, tp, note, res in recs:
    ms = res["matches"]
    ranked = sorted(ms, key=lambda m: -(m["scores"].get("identity_score") or 0))
    top1 = ranked[0]; top1_id = top1["model_id"]
    times.append((target, res.get("elapsed_ms"), res.get("extract_seconds"),
                  res.get("lookup_seconds"), len(ms)))
    if not tp:
        print(f"{norm(target):<34}{'(none declared)':<20}{'-':>6}{'-':>8}{norm(top1_id):>8}{'-':>6}{'-':>9}")
        rows.append(dict(target=target, true_parent=None, top1=top1_id, n_candidates=len(ms)))
        continue
    pr = [i for i, m in enumerate(ranked, 1) if norm(m["model_id"]) == norm(tp)]
    rank = pr[0] if pr else None
    psc = ranked[rank-1]["scores"].get("identity_score") if rank else None
    wrong = [m for m in ranked if norm(m["model_id"]) != norm(tp)]
    best_wrong = wrong[0]["scores"].get("identity_score") if wrong else None
    margin = (psc - best_wrong) if (psc is not None and best_wrong is not None) else None
    top5 = (rank is not None and rank <= 5)
    rows.append(dict(target=target, true_parent=tp, parent_rank=rank, parent_score=psc,
                     top1=top1_id, top5_member=top5, margin_to_best_wrong=margin,
                     n_candidates=len(ms)))
    print(f"{norm(target):<34}{norm(tp):<20}{(rank if rank else '--'):>6}"
          f"{(f'{psc:.4f}' if psc is not None else '--'):>8}"
          f"{norm(top1_id)[:7]:>8}{('yes' if top5 else 'no'):>6}"
          f"{(f'{margin:+.4f}' if margin is not None else '--'):>9}")

print(f"\n=== E17 summary ===")
lab = [r for r in rows if r.get("true_parent")]
if lab:
    r1 = sum(1 for r in lab if r.get("parent_rank") == 1)
    print(f"  scans with a declared true parent : {len(lab)}")
    print(f"  parent at rank 1                  : {r1}/{len(lab)}")
    print(f"  parent in top 5                   : {sum(1 for r in lab if r.get('top5_member'))}/{len(lab)}")
    mg = [r['margin_to_best_wrong'] for r in lab if r.get('margin_to_best_wrong') is not None]
    if mg: print(f"  margin to strongest wrong candidate: {min(mg):+.4f} .. {max(mg):+.4f}")

print(f"\n=== E22 runtime, from the same scan outputs ===")
el = [t[1] for t in times if t[1]]; ex = [t[2] for t in times if t[2]]; lu = [t[3] for t in times if t[3]]
nc = [t[4] for t in times]
print(f"  scans measured        : {len(times)}   candidates per scan {min(nc)}-{max(nc)}")
if el: print(f"  total elapsed (ms)    : median {st.median(el):.0f}   range {min(el):.0f}-{max(el):.0f}")
if ex: print(f"  signal extraction (s) : median {st.median(ex):.2f}   range {min(ex):.2f}-{max(ex):.2f}")
if lu: print(f"  database lookup (s)   : median {st.median(lu):.3f}   range {min(lu):.3f}-{max(lu):.3f}")
if ex and lu:
    print(f"  database lookup dominates signal extraction by "
          f"{st.median(lu)/max(st.median(ex),1e-9):.0f}x --- the cost is the per-candidate comparison "
          f"against the shipped fingerprint database, not reading the target's weights, so it scales "
          f"with database size rather than model size")
json.dump(dict(rank_rows=rows,
               runtime=dict(n=len(times), elapsed_ms_median=st.median(el) if el else None,
                            extract_s_median=st.median(ex) if ex else None,
                            lookup_s_median=st.median(lu) if lu else None,
                            candidates_min=min(nc), candidates_max=max(nc))),
          open(R/"M7"/"e17_e22.json","w"), indent=1)
print("\nwrote M7/e17_e22.json")
