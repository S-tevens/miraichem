#!/usr/bin/env bash
# Follow the README on a brand-new clone, as a newcomer would, and report what breaks.
# Usage: scripts/fresh_clone_check.sh [repo-url] [work-dir]
# Needs Linux/macOS/WSL with Python 3.11 and uv. Takes roughly 10 minutes (installs everything).
set -u
REPO="${1:-https://github.com/S-tevens/miraichem.git}"
WORK="${2:-/tmp/miraichem-fresh}"
PASS=0
FAIL=0
step() { printf '\n=== %s\n' "$1"; }
ok() { printf 'PASS  %s\n' "$1"; PASS=$((PASS + 1)); }
bad() { printf 'FAIL  %s\n' "$1"; FAIL=$((FAIL + 1)); }

rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK" || exit 2

step "1. clone"
git clone --quiet "$REPO" repo && cd repo && ok "git clone" || { bad "git clone"; exit 1; }
git log --oneline -1

step "2. environment (README: uv venv + uv pip install -r requirements.txt + editable install)"
uv venv --python 3.11 .venv >/dev/null 2>&1 && ok "uv venv" || bad "uv venv"
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q -r requirements.txt && ok "install requirements.txt" || bad "install requirements.txt"
uv pip install -q -e . --no-deps && ok "editable install" || bad "editable install"
command -v miraichem >/dev/null && ok "miraichem command on PATH" || bad "miraichem command on PATH"

step "3. README check: pytest -q (no PYTHONPATH tricks)"
if pytest -q 2>&1 | tail -3 | tee /tmp/fresh_pytest.txt | grep -q "passed"; then
  grep -q "failed\|error" /tmp/fresh_pytest.txt && bad "pytest has failures" || ok "pytest"
else
  bad "pytest did not complete"
fi
cat /tmp/fresh_pytest.txt

step "4. quick demo: sweep, rank"
miraichem sweep configs/sweeps/quick_h2.yaml --max-workers 4 >/tmp/fresh_sweep.txt 2>&1 \
  && ok "quick_h2 sweep" || { bad "quick_h2 sweep"; tail -5 /tmp/fresh_sweep.txt; }
miraichem rank --molecule h2 --backend noisy >/tmp/fresh_rank.txt 2>&1 \
  && ok "rank noisy" || { bad "rank noisy"; tail -5 /tmp/fresh_rank.txt; }

step "5. published results (no computation): rank, replicate, predictor"
miraichem rank --molecule h2 --backend noisy --results-dir results/published >/dev/null 2>&1 \
  && ok "rank on published results" || bad "rank on published results"
miraichem rank --molecule h2 --backend noisy --replicated \
  --runs-csv results/published/summary/replication_runs.csv >/dev/null 2>&1 \
  && ok "rank --replicated from the published CSV" || bad "rank --replicated from the published CSV"
miraichem predictor show --model results/published/models/error_predictor.json >/dev/null 2>&1 \
  && ok "predictor show (shipped model)" || bad "predictor show (shipped model)"

step "6. web app: API on the published results"
miraichem serve --port 8765 >/tmp/fresh_serve.txt 2>&1 &
SERVER=$!
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:8765/api/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8765/api/health >/dev/null && ok "server starts" || bad "server starts"
for path in "overview?source=published" "replicated?source=published" "devices?source=published" \
  "hardware?source=published" "curves?source=published" "predictor" "options" "sweeps"; do
  curl -sf -m 120 "http://127.0.0.1:8765/api/$path" >/dev/null && ok "GET /api/${path%%\?*}" || bad "GET /api/${path%%\?*}"
done
kill "$SERVER" 2>/dev/null

printf '\n=== summary: %s passed, %s failed (work dir: %s)\n' "$PASS" "$FAIL" "$WORK"
[ "$FAIL" -eq 0 ]
