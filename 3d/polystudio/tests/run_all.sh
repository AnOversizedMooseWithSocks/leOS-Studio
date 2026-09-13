#!/bin/sh
# Run every harness. The browser ones need a Chromium and a node_modules with puppeteer:
#   CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules sh tests/run_all.sh
# Skips the browser tests (with a notice) if those are not set.
cd "$(dirname "$0")/.." || exit 1
fail=0
run() { printf '\n=== %s ===\n' "$1"; shift; "$@" || fail=1; }

run "syntax"          sh -c 'node --check app.js && python3 -m py_compile backend.py app.py && echo "app.js and python compile"'
run "id_audit"        python3 tests/id_audit.py
run "scope_audit"     python3 tests/scope_audit.py
run "backend stubs"   python3 tests/undo_redo_test.py
run "render api (HTTP)" python3 tests/render_api_test.py
run "route sweep"     python3 tests/route_sweep.py

if [ -n "$CHROME" ] && [ -n "$PUPPETEER_PATH" ]; then
  for t in ui_smoke keymap_test touch_test dialog_test; do
    run "$t" node "tests/$t.js"
  done
  run "zip_roundtrip" sh -c 'node tests/zip_roundtrip.js && python3 -c "
import zipfile,sys
z=zipfile.ZipFile(\"/tmp/ps_turntable_test.zip\")
bad=z.testzip()
print(\"names:\", z.namelist(), \"| CRC check:\", \"OK\" if bad is None else bad)
sys.exit(0 if bad is None else 1)"'
else
  printf '\n(skipping browser tests: set CHROME and PUPPETEER_PATH to run them)\n'
fi

printf '\n'
if [ "$fail" = 0 ]; then echo "ALL HARNESSES PASSED"; else echo "SOMETHING FAILED"; fi
echo "NOTE: none of these run a real render. Run 'python3 quality_gate.py' with the engine installed"
echo "      before shipping anything that touches a render path."
exit $fail
