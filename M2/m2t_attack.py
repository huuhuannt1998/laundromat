"""M2t -- vocabulary remap / ID reorder.  The last unrun rung of design 9.1.

MECHANISM.  A consistent vocabulary reorder is function-preserving ON TEXT:
text -> new ids -> permuted embedding rows -> identical vectors, and the output
logits are the old logits permuted by the same map.  Cost is zero training
compute.  BPE merges are pairs of token STRINGS, so renumbering ids leaves them
untouched; only model.vocab and the embedding/lm_head row order change.

WHY IT MATTERS.  MPK's EAS reads K=64 anchor embedding rows selected BY INDEX
(get_anchor_ids).  Reordering the vocabulary puts different tokens in those
slots, so EAS should collapse -- the one thing the anchor-noise attack could
only buy at infinite perplexity.  The fraction-sweep decomposition says
destroying EAS *and* WVC jointly puts sigma_id at ~0.43, below the 0.5298 bar.
If this works, gate G1 moves from Exit 3 to Exit 1.

Special tokens are held fixed: config.json references bos/eos/pad by id.
"""
import os, sys, json, shutil, copy, subprocess, pathlib, glob
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES = pathlib.Path(str(_ROOT) + "/M2/results")
WORK = pathlib.Path(str(_ROOT) + "/M2/work"); WORK.mkdir(exist_ok=True)
PARENT = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
CHILD = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolLM2-135M-Instruct"
NULL = json.loads((RES / "null_frozen.json").read_text())["scores"]
import statistics as st

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

PI_CUT = st.quantiles(NULL, n=100)[74]


def mpk(args):
    p = subprocess.run(["uv", "run", "provenancekit", *args, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=5400,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}


PROBES = ["The professor's arguments were criticized for their",
          "In 1969, the first humans landed on the surface of",
          "The capital of France is Paris, and the capital of Germany is",
          "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot ="] * 6


def build_perm(vocab_size, fixed_ids, overlap, seed=17):
    """overlap = fraction of movable ids left in place."""
    rng = np.random.default_rng(seed)
    perm = np.arange(vocab_size)
    movable = np.array([i for i in range(vocab_size) if i not in fixed_ids])
    k = int(round((1.0 - overlap) * len(movable)))
    if k > 1:
        sel = rng.choice(movable, size=k, replace=False)
        shuf = sel.copy(); rng.shuffle(shuf)
        perm[sel] = shuf
    return perm


def run(overlap):
    tag = f"M2t-overlap{overlap:.2f}"
    out = WORK / tag.replace(".", "_")
    if out.exists(): shutil.rmtree(out)
    tok = AutoTokenizer.from_pretrained(CHILD)
    model = AutoModelForCausalLM.from_pretrained(CHILD).eval()
    ref = copy.deepcopy(model)

    tkjson = json.load(open(sorted(glob.glob(str(pathlib.Path(
        os.path.expanduser("~/.cache/huggingface/hub")) /
        ("models--" + CHILD.replace("/", "--")) / "snapshots" / "*" / "tokenizer.json")))[0]))
    vocab = tkjson["model"]["vocab"]
    fixed = {a["id"] for a in tkjson.get("added_tokens", [])}
    V = model.get_input_embeddings().weight.shape[0]
    perm = build_perm(V, fixed, overlap)            # perm[old_id] = new_id
    inv = np.argsort(perm)                          # inv[new_id] = old_id
    moved = int((perm != np.arange(V)).sum())

    # embedding rows: new row j must hold the old row inv[j]
    emb = model.get_input_embeddings().weight.data
    model.get_input_embeddings().weight.data = emb[torch.as_tensor(inv)].clone()
    tied = getattr(model.config, "tie_word_embeddings", False)
    if not tied and hasattr(model, "lm_head") and model.lm_head.weight.shape[0] == V:
        lw = model.lm_head.weight.data
        model.lm_head.weight.data = lw[torch.as_tensor(inv)].clone()

    # exactness on TEXT: new logits at perm[i] must equal old logits at i
    dev = []
    for s in PROBES[:8]:
        ids = tok(s, return_tensors="pt").input_ids
        old = ref(ids).logits[0, -1].float()
        newids = torch.as_tensor(perm[ids.numpy()])
        new = model(newids).logits[0, -1].float()
        dev.append((new[torch.as_tensor(perm)] - old).abs().max().item())
    max_dev = max(dev)

    model.save_pretrained(out)
    tok.save_pretrained(out)
    # rewrite the saved tokenizer's vocab with the new ids
    tp = out / "tokenizer.json"
    tj = json.load(open(tp))
    tj["model"]["vocab"] = {t: int(perm[i]) for t, i in vocab.items()}
    tj["truncation"] = None; tj["padding"] = None
    json.dump(tj, open(tp, "w"))
    del model, ref

    cmp_ = mpk(["compare", PARENT, str(out)])
    scan = mpk(["scan", str(out), "--top-k", "300", "--threshold", "0.0"])
    sc, sg = cmp_.get("scores", {}), cmp_.get("signals", {})
    ms = scan.get("matches", [])
    ps = PARENT.split("/")[-1]
    idx = [i for i, x in enumerate(ms, 1) if x["model_id"] == ps]
    rank = idx[0] if idx else None
    sp = sc.get("pipeline_score")
    pi = 100.0 * sum(1 for s in NULL if s <= sp) / len(NULL) if sp is not None else None
    E = 1 if (rank is not None and rank > 5 and pi is not None and pi <= 75) else 0
    shutil.rmtree(out, ignore_errors=True)
    rec = dict(name=tag, overlap=overlap, ids_moved=moved, frac_moved=moved / V,
               max_abs_logit_delta=max_dev, exact=bool(max_dev <= 1e-2),
               scores=sc, signals=sg, parent_rank=rank, pi=pi, E=E, kappa=0.0)
    print(f"{tag:<20} moved {moved:>6}/{V} tier={sc.get('mfi_tier')} "
          f"sig_id={sc.get('identity_score')} EAS={sg.get('eas')} END={sg.get('end')} "
          f"WVC={sg.get('wvc')} rank={rank} pi={pi} E={E} maxdlogit={max_dev:.2e} "
          f"{sc.get('provenance_decision')}", flush=True)
    return rec


if __name__ == "__main__":
    # guarded: this module is imported by decomp_transfer.py and fp32_verify.py for
    # mpk()/build_perm()/PROBES.  Without the guard the whole sweep re-ran on every
    # import -- deterministic, so results were never wrong, but it cost three full
    # MPK comparisons each time.
    fh = (RES / "m2t_attack.jsonl").open("w")
    print(f"pi* cut = {PI_CUT:.4f}   parent={PARENT}  child={CHILD}\n", flush=True)
    for ov in (0.9, 0.5, 0.0):
        try:
            fh.write(json.dumps(run(ov), default=str) + "\n"); fh.flush()
        except Exception as e:
            print(f"  overlap {ov} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    fh.close(); print("DONE-M2T", flush=True)
