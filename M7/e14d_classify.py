"""E14d -- classify E14b's key-less configurations by a written rule, and recompute the rate.

M7/e14b_classified.json split E14b's 47 configurations lacking `architectures` into 23 GGUF
repositories, 6 test fixtures and 18 "genuine" models, but no script records the rule, and review
round 5 (N9) noted that one of the 18, robot-test/dummy-tokenizer-fast-with-model-config, looks
like a fixture itself. Section VIII's criterion is "setting aside ... GGUF repositories the tool
does not read and ... test fixtures". This applies that criterion as a written rule:

  GGUF          repository name ends in "-GGUF" (the tool reads HF-format weights only);
  test fixture  a repository published for library tests rather than use: a namespace that is a
                test account (…-testing, robot-test, peft-internal-testing) or a name marking a
                random or dummy artifact (tiny-random, dummy).

robot-test/dummy-tokenizer-fast-with-model-config meets the rule on both counts, and its hub
listing (fetched 2026-09-25) holds no weight file at all: .gitattributes, config.json,
special_tokens_map.json, tokenizer.json, tokenizer_config.json. Its config.json is
{"model_type": "albert", "tokenizer_class": "PreTrainedTokenizerFast"}.

Reads M7/e14b_hub_prevalence.jsonl. Writes a NEW file, M7/e14d_classified.json; e14b_classified.json
is left as it was.
"""
import json, re
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

TEST_NS = re.compile(r"(^|-)testing$|^robot-test$|^peft-internal-testing$")
TEST_NAME = re.compile(r"tiny-random|dummy", re.I)

rows = [json.loads(l) for l in (_ROOT/"M7"/"e14b_hub_prevalence.jsonl").read_text().splitlines() if l.strip()]
fetched = [r for r in rows if r.get("fetched")]
keyless = [r for r in fetched if (not r["has_architectures"]) or r["architectures_null"]]

def kind(repo):
    ns, name = repo.split("/", 1)
    if name.upper().endswith("-GGUF"): return "gguf"
    if TEST_NS.search(ns) or TEST_NAME.search(name): return "test fixture"
    return "ordinary"

groups = {}
for r in keyless: groups.setdefault(kind(r["repo"]), []).append(r["repo"])
old = json.loads((_ROOT/"M7"/"e14b_classified.json").read_text())
n, k = len(fetched), len(groups.get("ordinary", []))
out = dict(rule=dict(gguf="name ends in -GGUF",
                     test_fixture="test-account namespace (…-testing, robot-test, peft-internal-testing) "
                                  "or name containing tiny-random / dummy"),
           n_fetched=n, n_keyless=len(keyless),
           counts={g: len(v) for g, v in groups.items()}, groups=groups,
           rate_ordinary=k / n,
           changed_from_e14b=sorted(set(old["genuine"]) - set(groups.get("ordinary", []))))
(_ROOT/"M7"/"e14d_classified.json").write_text(json.dumps(out, indent=1))
print({g: len(v) for g, v in groups.items()}, f"ordinary {k}/{n} = {100*k/n:.2f}%")
print("reclassified from e14b's genuine set:", out["changed_from_e14b"])
print("wrote M7/e14d_classified.json")
