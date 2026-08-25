"""E15 -- one consolidated transform x model matrix over every attack arm we ran.

The review asks for a single place a reviewer can see every transform against every
model with its fidelity and its verdict, instead of the per-arm files. Built from
frozen M2 results only. Emitted as CSV + Markdown for the artifact; the paper's page
budget is full, so the paper points here rather than reprinting it.
"""
import json, glob, pathlib, csv

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R = pathlib.Path(_ROOT)
OUT_CSV = R/"M7"/"e15_transform_matrix.csv"
OUT_MD  = R/"M7"/"e15_transform_matrix.md"

rows = []
for f in sorted(glob.glob(str(R/"M2"/"results"/"*.jsonl"))):
    base = pathlib.Path(f).stem
    if base.startswith(("null_", "wide_scans")): continue
    for line in open(f):
        if not line.strip(): continue
        try: r = json.loads(line)
        except: continue
        sc = r.get("scores") or r.get("cmp") or {}
        if not isinstance(sc, dict) or sc.get("identity_score") is None: continue
        fid = r.get("fidelity") or {}
        applied = r.get("applied") or r.get("name") or r.get("arm") or base
        if isinstance(applied, list):
            applied = "+".join(a.get("transform", str(a)) if isinstance(a, dict) else str(a)
                               for a in applied) or "none"
        rows.append(dict(
            arm=base, transform=str(applied)[:60],
            parent=r.get("parent",""), child=r.get("child") or r.get("model",""),
            identity=sc.get("identity_score"), pipeline=sc.get("pipeline_score"),
            tier=sc.get("mfi_tier"), verdict=sc.get("provenance_decision"),
            max_dlogit=fid.get("max_abs_logit_delta"),
            rel_ppl=fid.get("rel_ppl_delta"),
            exact=fid.get("exact_within_tolerance"),
        ))
rows.sort(key=lambda r: (r["arm"], -(r["identity"] or 0)))
with OUT_CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def cap(r):
    """Capability verdict. Absence of a fidelity record is NOT evidence of preservation:
    it is reported as 'unmeasured' and excluded from the preserving tally, because
    counting it as preserved would manufacture an evasion that was never demonstrated."""
    if r["exact"] is True: return "exact"
    d = r["max_dlogit"]
    if d is None: return "unmeasured"
    return "preserved" if d <= 1e-2 else f"BROKEN ({d:.3g})"

lines = ["# E15 — consolidated transform matrix", "",
         f"{len(rows)} arms over {len({r['arm'] for r in rows})} experiment files. "
         "`capability` is the predeclared criterion: max|dlogit| <= 1e-2 in float32 on a fixed probe set.",
         "", "| arm | transform | child | sig_id | pipeline | tier | verdict | capability |",
         "|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['arm']} | {r['transform']} | {(r['child'] or '').split('/')[-1]} | "
                 f"{r['identity']} | {r['pipeline']} | {r['tier']} | {r['verdict']} | {cap(r)} |")
brk = [r for r in rows if cap(r).startswith("BROKEN")]
evade = [r for r in rows if (r["identity"] or 1) < 0.5298]
lines += ["", f"- arms breaking the capability criterion: **{len(brk)}**",
          f"- arms with identity score below the 0.5298 evasion bar: **{len(evade)}**",
          f"- of those, demonstrably capability-preserving: "
          f"**{len([r for r in evade if cap(r) in ('exact','preserved')])}**",
          f"- of those, capability unmeasured in this arm's file: "
          f"**{len([r for r in evade if cap(r)=='unmeasured'])}**"]
OUT_MD.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT_CSV.name} and {OUT_MD.name}: {len(rows)} arms")
print(f"  capability-broken arms         : {len(brk)}")
print(f"  arms below the evasion bar     : {len(evade)}")
print(f"  ... demonstrably capability-preserving : {len([r for r in evade if cap(r) in ('exact','preserved')])}")
print(f"  ... capability unmeasured in-file      : {len([r for r in evade if cap(r)=='unmeasured'])}")
