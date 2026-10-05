"""E46 -- recompute the per-signal real-model minima and the floor/subset arithmetic from named files.

Why. CORRECTIONS.md #21 printed a "real published models" column (anchor 0.5570, embedding-norm
0.0002, layer-energy 0.0395; WVC 0.0002, NLF 0.0000) described as minima over "1,100+" or "1,104"
genuine comparisons, but no script on disk produces it (2026-10-02 audit). This recomputes it.

Sources (read only):
  M2/results/signal_floors.jsonl  transform floors: each signal driven by a targeted perturbation
                                  with the other four at 1.0 (SmolLM2-135M against itself), plus the
                                  all-signal arm
  M2/results/wide_scans.jsonl     1,400 real comparisons: 10 targets x 140 reference-catalogue
                                  assets, signals nested under result.matches[].scores
  M3/verify_claims.jsonl          24 declared parent-child pairs (for the WVC 0.0002 cell)
  M5/sweep_bert.jsonl             the BERT sweep as served (for the NLF 0.0000 cells; .bin-only
                                  children, a serialisation-order artefact per E35-E39)
Weights from the pinned verifier, M1/oracle/model-provenance-kit/src/provenancekit/config/constants.py.
Evasion bar 0.5298 (null p75; analysis/prereg_stats.json). Writes a NEW file,
M2/results/e46_floor_minima.json.
"""
import json, itertools, pathlib, os
R = pathlib.Path(os.environ.get("LAUNDROMAT_ROOT",
    next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())))
W = {"eas": 0.36, "wvc": 0.21, "end": 0.19, "lep": 0.16, "nlf": 0.08}
BAR = 0.5298
S = list(W)

arms = {json.loads(l)["arm"]: json.loads(l) for l in (R/"M2/results/signal_floors.jsonl").read_text().splitlines() if l.strip()}
def sig(a):
    s = arms[a]["signals"]; return s if isinstance(s, dict) else json.loads(s.replace("'", '"'))
transform = {k: sig(k.upper())[k] for k in S}
all_arm = dict(sigma=arms["ALL"]["sigma_id"], signals=sig("ALL"), tier=arms["ALL"]["tier"],
               pipeline=arms["ALL"]["pipeline"], verdict=arms["ALL"]["verdict"])
pred_all = sum(W[k] * transform[k] for k in S)

ws = [json.loads(l) for l in (R/"M2/results/wide_scans.jsonl").read_text().splitlines() if l.strip()]
comps = [(r["target"], m["model_id"], m["scores"]) for r in ws for m in r["result"]["matches"]]
wide = {k: min(c[2][k] for c in comps if c[2].get(k) is not None) for k in S}
wide_n = {k: sum(1 for c in comps if c[2].get(k) is not None) for k in S}
wide_arg = {k: [(c[0], c[1]) for c in comps if c[2].get(k) == wide[k]][:3] for k in S}

vc = [json.loads(l) for l in (R/"M3/verify_claims.jsonl").read_text().splitlines() if l.strip()]
vc_wvc = min((r for r in vc if (r.get("signals") or {}).get("wvc") is not None), key=lambda r: r["signals"]["wvc"])
sb = [json.loads(l) for l in (R/"M5/sweep_bert.jsonl").read_text().splitlines() if l.strip()]
sb_nlf0 = [r["child"] for r in sb if (r.get("signals") or {}).get("nlf") == 0.0]

def analysis(floor):
    lo = {k: min(transform[k], floor[k]) for k in S}
    least = sum(W[k] * lo[k] for k in S)
    subs = {}
    for r in (1, 2, 3, 4, 5):
        for c in itertools.combinations(S, r):
            subs["+".join(c)] = round(1 - sum(W[k] * (1 - lo[k]) for k in c), 6)
    pairs = {k: v for k, v in subs.items() if k.count("+") == 1}
    triples = {k: v for k, v in subs.items() if k.count("+") == 2}
    return dict(lower_of_two=lo, least_identity=round(least, 6), anchor_mass=round(W["eas"] * lo["eas"], 6),
                pairs_below_bar=sorted(k for k, v in pairs.items() if v < BAR),
                best_pair=min(pairs.items(), key=lambda kv: kv[1]),
                eas_wvc=pairs["eas+wvc"],
                triples_below_bar=sorted(((k, v) for k, v in triples.items() if v < BAR), key=lambda kv: kv[1]),
                triples_sorted=sorted(triples.items(), key=lambda kv: kv[1]),
                subsets=subs)

wide_only = analysis(wide)
with_qualified = analysis({**wide, "wvc": min(wide["wvc"], vc_wvc["signals"]["wvc"]), "nlf": 0.0})
out = dict(weights=W, bar=BAR, transform_floors=transform, all_arm=all_arm,
           all_arm_predicted_from_single_arms=round(pred_all, 6),
           all_arm_residual=round(pred_all - all_arm["sigma"], 6),
           wide_scans=dict(n_comparisons=len(comps), n_targets=len(ws), observations_per_signal=wide_n,
                           minima=wide, examples=wide_arg),
           qualified_cells=dict(
               wvc=dict(value=vc_wvc["signals"]["wvc"], pair=[vc_wvc.get("child"), vc_wvc.get("declared_parent")],
                        source="M3/verify_claims.jsonl"),
               nlf=dict(value=0.0, children=sb_nlf0, source="M5/sweep_bert.jsonl",
                        caveat=".bin-only children; serialisation-order artefact (E35-E39)")),
           wide_scans_only=wide_only, with_qualified_cells=with_qualified)
(R/"M2/results/e46_floor_minima.json").write_text(json.dumps(out, indent=1))
print("transform floors", transform, "ALL", all_arm["sigma"], "pred", round(pred_all, 4))
print("wide_scans", len(comps), "comparisons; minima", wide, "n", wide_n)
print("qualified: wvc", vc_wvc["signals"]["wvc"], vc_wvc.get("child"), "| nlf 0.0 children", sb_nlf0)
for name, a in (("wide-scans only", wide_only), ("with qualified cells", with_qualified)):
    print(f"{name}: least {a['least_identity']}, anchor {a['anchor_mass']}, pairs below bar {a['pairs_below_bar']}, "
          f"best pair {a['best_pair']}, eas+wvc {a['eas_wvc']}, triples below bar {a['triples_below_bar']}")
