#!/bin/bash
# Sequential experiment queue for LAUNDROMAT. NEVER run these in parallel: each 6.9B
# arm holds one fp16 model (~14 GB) and the laptop has 24 GB total. The machine
# crashed once with several projects running at the same time.
#
#   ./run_queue.sh            preflight only, then list what would run
#   ./run_queue.sh --go       preflight, then run the queue sequentially
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

# PATH restoration. After a restart this shell's PATH lost /opt/homebrew/bin and
# the user bin dirs, which silently removed pdftotext, then python3-with-torch,
# then uv (which MPK's `uv run provenancekit` needs). Restore them up front so the
# queue does not fail one binary at a time.
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:$PATH"
for t in uv pdftotext; do
  command -v "$t" >/dev/null || { echo "FATAL: $t not found on PATH"; exit 1; }
done
echo "uv: $(command -v uv)"

# Resolve an interpreter that actually has the ML stack. After a restart, PATH's
# python3 became system 3.9 (no transformers) -- the same class of quiet breakage
# that took out pdftotext. Fail loudly rather than half-run.
PY_ML=""
for c in "$HOME/miniconda3/bin/python3" /opt/homebrew/bin/python3 python3; do
  if "$c" -c "import transformers,torch,scipy,numpy" 2>/dev/null; then PY_ML="$c"; break; fi
done
if [ -z "$PY_ML" ]; then
  echo "FATAL: no python3 with transformers+torch+scipy+numpy found."; exit 1
fi
echo "interpreter: $PY_ML"

echo "=== PREFLIGHT ==="
"$PY_ML" preflight.py || { echo; echo "Refusing to start. Close other projects, then re-run."; exit 1; }

echo
echo "=== QUEUE ==="
echo "  1. E27  ancestor control at 6.9B   (resumable; weights already materialised)"

if [ "${1:-}" != "--go" ]; then
  echo
  echo "Dry run. Pass --go to execute."
  exit 0
fi

echo
echo "=== 1/1  E27 ==="
"$PY_ML" e27_ancestor_scale.py 2>&1 | tee -a e27_run.log
echo
echo "=== RESULT ==="
"$PY_ML" - <<'PY'
import json, pathlib
p = pathlib.Path("e27_ancestor_scale.jsonl")
if not p.exists() or not p.read_text().strip():
    print("  no record written -- check e27_run.log"); raise SystemExit
for l in p.open():
    d = json.loads(l); s = d["scores"]
    print(f"  step{d['step']}: identity={s['identity_score']:.4f} tier={s['mfi_tier']} "
          f"{s['provenance_decision']}  LAP={d['lap_defence']:.4f}")
print("  compare against the E23 same-recipe FALSE POSITIVE at 6.9B: identity 0.8211, LAP 0.3250")
PY
