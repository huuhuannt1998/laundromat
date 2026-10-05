# LAUNDROMAT artifact entry points. See ARTIFACT.md.
PY := python3
ROOT := $(shell pwd)
# Poppler is not always on PATH (it lives in /opt/homebrew/bin under Homebrew and
# dropped off PATH once mid-project, which silently gutted check-paper). Resolve it
# explicitly and fail loudly rather than reporting a partial result.
PDFLATEX  := $(shell command -v pdflatex 2>/dev/null || echo /Library/TeX/texbin/pdflatex)
PDFTOTEXT := $(shell command -v pdftotext 2>/dev/null || echo /opt/homebrew/bin/pdftotext)
PDFINFO   := $(shell command -v pdfinfo   2>/dev/null || echo /opt/homebrew/bin/pdfinfo)
# bibtex must run inside check-paper. Without it the gate reports page counts for a PDF built
# against a STALE main.bbl: adding eleven references once read as "0 undefined cites, 18 pages"
# while the real, re-bibtex'd document was 19.
BIBTEX    := $(shell command -v bibtex 2>/dev/null || echo /Library/TeX/texbin/bibtex)
MPK := $(ROOT)/M1/oracle/model-provenance-kit

.PHONY: help check-paper verify-data smoke reproduce-figures reproduce-tables reproduce-manifest \
        reproduce-ladder reproduce-awm reproduce-substrate reproduce-defence check-version \
        reproduce-recalibration reproduce-stats reproduce-deployment

help:
	@echo "check-paper        build the manuscript and read the REAL warnings from main.log"
	@echo "verify-data        check the 80 frozen result files against MANIFEST-dataintegrity.txt"
	@echo "smoke              ~2 min: one gate comparison + one figure regeneration"
	@echo "check-version      confirm the pinned verifier commit is still upstream HEAD"
	@echo "reproduce-manifest rebuild the gold lineage manifest from frozen inputs"
	@echo "reproduce-figures  regenerate all four figures from frozen data"
	@echo "reproduce-tables   print every table's numbers from its frozen source"
	@echo "reproduce-ladder   E12: nine pythia-160m checkpoints vs the final one (hours)"
	@echo "reproduce-awm      E3: official AWM baseline on the causal subset (minutes)"
	@echo "reproduce-substrate E4: black-box tester at 2000 prompts (hours)"
	@echo "reproduce-defence  E18: expanded alignment defence, family-disjoint holdout (hours)"
	@echo "reproduce-recalibration E9: refit the combination on audited labels (seconds)"
	@echo "reproduce-deployment E25: verdict -> ML-BOM -> OPA gate (needs opa on PATH)"
	@echo "reproduce-stats    E13/E15/E17/E22: bootstrap CIs, transform matrix, rank, runtime (seconds)"

# NOTE: the body-end gate below greps the PDF for the LAST SENTENCE of the conclusion.
# Editing that sentence silently turns the gate into 'ANCHOR NOT FOUND'. If you reword it,
# update the regex in the same commit. Current anchor: "predates the tool".
check-paper:
	@test -x "$(PDFLATEX)" || { echo "  FATAL: pdflatex not found ($(PDFLATEX)) -- CANNOT REBUILD, gate would read a stale PDF"; exit 1; }
	@test -x "$(PDFTOTEXT)" || { echo "  FATAL: pdftotext not found ($(PDFTOTEXT)) -- gate cannot run"; exit 1; }
	@test -x "$(PDFINFO)"   || { echo "  FATAL: pdfinfo not found ($(PDFINFO)) -- gate cannot run"; exit 1; }
	@test -x "$(BIBTEX)"    || { echo "  FATAL: bibtex not found ($(BIBTEX)) -- gate would read a stale bibliography"; exit 1; }
	@cd manuscripts/laundromat && \
	  { $(PDFLATEX) -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1 && \
	    $(BIBTEX) main >/dev/null 2>&1; \
	    $(PDFLATEX) -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1 && \
	    $(PDFLATEX) -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1; } \
	    || { echo "  FATAL: pdflatex failed -- refusing to report gates on a stale PDF"; exit 1; }; \
	  echo "  (warnings are read from main.log, NOT latexmk stdout -- latexmk's stdout"; \
	  echo "   contains no LaTeX warnings, which silently voided this check once)"; \
	  printf "  undefined refs   %s\n"  "$$(grep -c 'Warning: Reference' main.log)"; \
	  printf "  undefined cites  %s\n"  "$$(grep -c 'Warning: Citation' main.log)"; \
	  printf "  errors           %s\n"  "$$(grep -cE '^! |^\./[^:]+:[0-9]+: ' main.log)"; \
	  printf "  overfull         %s\n"  "$$(grep -c Overfull main.log)"; \
	  printf "  '??' in PDF      %s\n"  "$$($(PDFTOTEXT) main.pdf - 2>/dev/null | grep -c '??')"; \
	  $(PY) -c "import re,subprocess; t=subprocess.run(['$(PDFTOTEXT)','main.pdf','-'],capture_output=True,text=True).stdout; pages=t.split(chr(12)); hits=[i+1 for i,p in enumerate(pages) if re.search(r'predates the tool', re.sub(r'\s+',' ',p))]; print('  body text ends   p%s (limit 13)'%(hits[-1] if hits else '?? ANCHOR NOT FOUND'))"; \
	  $(PDFINFO) main.pdf | awk '/Pages/{printf "  total pages      %s (limit 18)\n",$$2}'

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
	    || echo "MOVED - re-run the replication and update section IV"

smoke:
	@echo "--- gate self-comparison (expect Confirmed Match) ---"
	@cd $(MPK) && uv run provenancekit compare gpt2 gpt2 --json --no-cache 2>/dev/null \
	  | $(PY) -c "import sys,json; s=sys.stdin.read(); d=json.loads(s[s.find('{'):]); \
print('  pipeline', d['scores']['pipeline_score'], '|', d['scores']['provenance_decision'])"
	@echo "--- figure 3 regeneration (expect 0.8692 / -0.3327 / 0.6927 / 0.8295) ---"
	@$(PY) M6/make_fig3.py

reproduce-manifest:
	$(PY) M6/build_gold_manifest.py

reproduce-figures:
	$(PY) M6/make_fig0.py && $(PY) M6/make_fig2.py && $(PY) M6/make_fig3.py && $(PY) M6/make_fig4.py
	@echo "  (make_fig1.py retired: fig0 merged its tier logic into the section I architecture figure)"

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
