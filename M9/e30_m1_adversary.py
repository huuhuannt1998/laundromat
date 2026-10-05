"""E30 -- what an M1 adversary (publisher-written metadata only) buys, in both directions.

Section III places two findings at control level M1 but neither experiment had an adversary
WRITE a false configuration: the gate override (V-E) was measured on unmodified configurations
plus our own repair, and the metadata matrix (E7) varied fields on a true pair. This closes
that gap with weights held byte-identical throughout.

(a) FORGE descent. nlpaueb/legal-bert-base-uncased is pretrained from scratch (card: "trained
    from scratch in the legal corpora using a newly created vocabulary"); the verifier reads it
    at tier 3 and returns Not Matched at identity 0.6061 against bert-base-uncased. Its
    dimensions already match bert-base-uncased; the only Tier-1 hash field that differs is
    `architectures` (["BertForPreTraining"] vs ["BertForMaskedLM"]). The adversary rewrites that
    one value. Tier-1 hash fields (metadata.py `_derive_hashes`): model_type, architectures,
    hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    intermediate_size, vocab_size, max_position_embeddings.

(b) DENY descent. A true child renames `model_type` to a different REGISTERED type, keeping the
    tokenizer loadable through `tokenizer_class` in tokenizer_config.json. model_type enters
    both hashes, so tiers 1 and 2 cannot fire and the verdict falls to the weight signals.
    Two children: textattack/bert-base-uncased-SST-2 (tier 2 as published, identity 0.8249)
    and cardiffnlp/twitter-roberta-base (tier 1 as published, identity 0.6899).

Every arm is run through the pinned CLI with --no-cache; unmodified copies are run as
controls. Output: M9/e30_m1_adversary.jsonl.
"""
import os, json, shutil, pathlib, subprocess, glob
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e30"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = ROOT / "M9" / "e30_m1_adversary.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))

def snapshot(mid):
    """The cached snapshot directory holding config.json (weights may sit in a sibling snapshot)."""
    base = HUB / ("models--" + mid.replace("/", "--")) / "snapshots"
    cfgs = [p for p in base.glob("*/config.json")]
    assert cfgs, f"no cached config for {mid}"
    return cfgs[0].parent, base

def materialise(mid, tag):
    """Copy every file of the model into WORK/tag (resolving symlinks) so the config can be edited."""
    out = WORK / tag
    if (out/"config.json").exists(): return out
    out.mkdir(parents=True)
    snap, base = snapshot(mid)
    seen=set()
    for s in sorted(base.glob("*")):
        for f in s.iterdir():
            if f.name in seen or f.is_dir(): continue
            if f.name.endswith((".msgpack",".h5",".meta",".index")) or "ckpt" in f.name: continue
            shutil.copy(f.resolve(), out/f.name); seen.add(f.name)
    return out

def edit_json(path, updates):
    d = json.loads(path.read_text()) if path.exists() else {}
    d.update(updates); path.write_text(json.dumps(d, indent=1))

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", str(a), str(b), "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    if i >= 0: return json.loads(p.stdout[i:]), None
    return None, (p.stderr or "")[-400:]

ARMS = [
 # tag, direction, child, parent, config edits, tokenizer_config edits, note
 ("forge-legalbert-control", "forge", "nlpaueb/legal-bert-base-uncased", "google-bert/bert-base-uncased",
  {}, {}, "unmodified copy; expect tier 3, Not Matched, identity 0.6061"),
 ("forge-legalbert-architectures", "forge", "nlpaueb/legal-bert-base-uncased", "google-bert/bert-base-uncased",
  {"architectures": ["BertForMaskedLM"]}, {}, "one field rewritten to the target's value; every Tier-1 hash field now matches"),
 ("deny-sst2-control", "deny", "textattack/bert-base-uncased-SST-2", "google-bert/bert-base-uncased",
  {}, {}, "unmodified copy; expect tier 2, Confirmed, identity 0.8249"),
 ("deny-sst2-modeltype-roberta", "deny", "textattack/bert-base-uncased-SST-2", "google-bert/bert-base-uncased",
  {"model_type": "roberta"}, {"tokenizer_class": "BertTokenizer"}, "registered but different model_type; tokenizer kept loadable"),
 ("deny-sst2-modeltype-electra", "deny", "textattack/bert-base-uncased-SST-2", "google-bert/bert-base-uncased",
  {"model_type": "electra"}, {"tokenizer_class": "BertTokenizer"}, "second registered type, same field set as bert"),
 ("deny-twitter-control", "deny", "cardiffnlp/twitter-roberta-base", "FacebookAI/roberta-base",
  {}, {}, "unmodified copy; expect tier 1, Confirmed, identity 0.6899"),
 ("deny-twitter-modeltype-bert", "deny", "cardiffnlp/twitter-roberta-base", "FacebookAI/roberta-base",
  {"model_type": "bert"}, {"tokenizer_class": "RobertaTokenizer"}, "registered but different model_type; tokenizer kept loadable"),
]

done=set()
if OUT.exists():
    for l in OUT.open():
        if l.strip(): done.add(json.loads(l)["tag"])
with OUT.open("a") as fh:
    for tag, direction, child, parent, cfg_edit, tok_edit, note in ARMS:
        if tag in done: print(f"[skip] {tag}"); continue
        d = materialise(child, tag)
        if cfg_edit: edit_json(d/"config.json", cfg_edit)
        if tok_edit: edit_json(d/"tokenizer_config.json", tok_edit)
        r, err = mpk(d, parent)
        rec = {"tag": tag, "direction": direction, "child": child, "parent": parent,
               "config_edits": cfg_edit, "tokenizer_config_edits": tok_edit, "note": note,
               "weights_modified": False}
        if r is None:
            rec.update(no_verdict=True, error_tail=err)
            print(f"{tag:34} NO VERDICT  {err[-160:]!r}")
        else:
            s, g = r["scores"], r["signals"]
            rec.update(no_verdict=False, scores=s, signals=g)
            print(f"{tag:34} tier {s['mfi_tier']} ({s['mfi_match']})  pipe {s['pipeline_score']:.4f}  "
                  f"identity {s['identity_score']}  wvc {g.get('wvc')}  -> {s['provenance_decision']}")
        fh.write(json.dumps(rec)+"\n"); fh.flush()
        shutil.rmtree(d, ignore_errors=True)
print("DONE-E30")
