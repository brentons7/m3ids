#!/bin/bash
# Run a job file: one line of `run.py experiment` flags per job, blank lines and #comments skipped.
# Several devices can share one file: each runs every N-th job, starting at its own WORKER index.
#   bash scripts/run_queue.sh JOBFILE [WORKER N]      e.g. nano1: ... 0 2    nano2: ... 1 2
# A failed job is logged and skipped; the queue keeps going. Logs: logs/<jobfile>_<line>.log
cd "$(dirname "$0")/.."
jobfile="$1"; worker="${2:-0}"; n="${3:-1}"
if [ ! -f "$jobfile" ]; then echo "usage: bash scripts/run_queue.sh JOBFILE [WORKER N]"; exit 1; fi
if ! python3 -c "import torch" 2> /dev/null; then
    echo "python3 can't import torch. Activate the venv first: source .venv/bin/activate"; exit 1
fi
mkdir -p logs
name=$(basename "$jobfile" .txt)

i=0; lineno=0
while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    if (( i % n == worker )); then
        log="logs/${name}_${lineno}.log"
        echo "$(date +%H:%M) start line $lineno: $line"
        # shellcheck disable=SC2086
        if python3 -u run.py experiment $line < /dev/null > "$log" 2>&1; then
            echo "$(date +%H:%M) done  line $lineno"
        else
            echo "$(date +%H:%M) FAILED line $lineno (see $log)"
        fi
    fi
    i=$((i + 1))
done < "$jobfile"
echo "$(date +%H:%M) queue finished"
