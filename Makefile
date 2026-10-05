# LAUNDROMAT artifact entry points. See README.md.
PY := python3
ROOT := $(shell pwd)
MPK := $(ROOT)/M1/oracle/model-provenance-kit

.PHONY: help verify-data smoke reproduce-figures reproduce-tables reproduce-manifest \
        reproduce-ladder reproduce-awm reproduce-substrate reproduce-defence check-version \
        reproduce-recalibration reproduce-stats reproduce-deployment

help:
	@echo "verify-data        check the frozen result files against MANIFEST-dataintegrity.txt"
	@echo "smoke              ~2 min: one verifier self-comparison"
	@echo "check-version      confirm the pinned verifier commit is still upstream HEAD"
	@echo "reproduce-manifest rebuild the gold lineage manifest from frozen inputs"
	@echo "reproduce-figures  regenerate the two data figures from frozen data"
	@echo "reproduce-tables   print every table's numbers from its frozen source"
	@echo "reproduce-ladder   E12: nine pythia-160m checkpoints vs the final one (hours)"
	@echo "reproduce-awm      E3: official AWM baseline on the causal subset (minutes)"
	@echo "reproduce-substrate E4: black-box tester at 2000 prompts (hours)"
	@echo "reproduce-defence  E18: expanded alignment defence, family-disjoint holdout (hours)"
	@echo "reproduce-recalibration E9: refit the combination on audited labels (seconds)"
	@echo "reproduce-deployment E25: verdict -> ML-BOM -> OPA gate (needs opa on PATH)"
	@echo "reproduce-stats    E13/E15/E17/E22: bootstrap CIs, transform matrix, rank, runtime (seconds)"

verify-data:
	@$(PY) -c "import hashlib,os,sys; \
ok=bad=0; \
lines=[l.strip() for l in open('MANIFEST-dataintegrity.txt') if l.strip() and not l.startswith('#')]; \
[ (lambda md5,p: (globals().__setitem__('ok',globals()['ok']+1) if os.path.exists(p) and hashlib.md5(open(p,'rb').read()).hexdigest()==md5 else (print('CHANGED/MISSING',p), globals().__setitem__('bad',globals()['bad']+1))))(l.split()[0], ' '.join(l.split()[1:]).lstrip('*')) for l in lines ]; \
print(f'frozen: {ok} unchanged, {bad} changed'); sys.exit(1 if bad else 0)"

check-version:
	@cd $(MPK) && git fetch -q origin && \
	  echo "pinned : $$(git rev-parse HEAD)" && \
	  echo "upstream: $$(git rev-parse origin/main)" && \
	  test "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" \
	    && echo "MATCH - the paper's attestation still holds" \
	    || echo "MOVED - re-run the replication and update the version attestation"

smoke:
	@echo "--- gate self-comparison (expect Confirmed Match) ---"
	@cd $(MPK) && uv run provenancekit compare gpt2 gpt2 --json --no-cache 2>/dev/null \
	  | $(PY) -c "import sys,json; s=sys.stdin.read(); d=json.loads(s[s.find('{'):]); \
print('  pipeline', d['scores']['pipeline_score'], '|', d['scores']['provenance_decision'])"

reproduce-manifest:
	$(PY) M6/build_gold_manifest.py

reproduce-figures:
	$(PY) M6/make_fig2.py && $(PY) M6/make_fig4.py

reproduce-tables:
	$(PY) analysis/print_tables.py

reproduce-ladder:
	$(PY) M6/e12_pythia_ladder.py

reproduce-awm:
	$(PY) M7/awm/e3_awm_baseline.py && $(PY) M7/awm/e3_analysis.py

reproduce-substrate:
	cd M1/oracle/model_provenance_testing && $(PY) tester.py --prompt_id -1 \
	  --file_parents e4_parents.txt --file_candidates e4_candidates.txt \
	  --no_prompts 2000 --device mps

reproduce-defence:
	$(PY) M7/e18_defence.py && $(PY) M7/e18_analysis.py

reproduce-recalibration:
	$(PY) M7/e9_recalibrate.py

reproduce-deployment:
	$(PY) M8/run_testbed.py

reproduce-stats:
	$(PY) M7/e13_regression.py && $(PY) M7/e17_rank_e22_runtime.py && $(PY) M7/e15_transform_matrix.py
