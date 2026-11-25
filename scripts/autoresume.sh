#!/usr/bin/env bash
# Resume a triage run that paused because the metered endpoint's quota ran out.
#
# A budget pause is deliberate, not a failure: progress is saved and the run
# waits for a human to press Resume (processing.py:278). That is the right
# behaviour for a person watching, and the wrong one for a corpus that has to
# be finished by a deadline -- left alone it sits idle all night and loses the
# hours the quota has already refilled.
#
# So: poll, and re-POST start-processing whenever the run is paused on budget
# and work remains. Exits on its own once nothing is left, so it cannot outlive
# the job it was started for.
#
#   scripts/autoresume.sh <project_id> [base_url] [poll_seconds]
#   tail -f data/logs/autoresume-<project_id>.log
#   pkill -f 'autoresume.sh <project_id>'      # stop it
set -uo pipefail

PROJECT="${1:?usage: autoresume.sh <project_id> [base_url] [poll_seconds]}"
BASE="${2:-http://localhost:8000}"
POLL="${3:-300}"
LOG_DIR="${DA_LOG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/logs}"
LOG="$LOG_DIR/autoresume-$PROJECT.log"
mkdir -p "$LOG_DIR"

say() { printf '%s  %s\n' "$(date -Is)" "$*" >>"$LOG"; }

say "watching project $PROJECT at $BASE every ${POLL}s (pid $$)"

while :; do
    progress="$(curl -sS --max-time 30 "$BASE/api/projects/$PROJECT/processing-progress" 2>>"$LOG")"
    if [ -z "$progress" ]; then
        say "no answer from $BASE -- will retry"
        sleep "$POLL"; continue
    fi

    read -r running paused remaining note <<<"$(printf '%s' "$progress" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d.get("running"), d.get("paused_on_budget"),
      d.get("remaining_urls"), (d.get("budget_note") or "-").replace(" ", "_"))
')" || { say "could not parse progress"; sleep "$POLL"; continue; }

    if [ "$remaining" = "0" ]; then
        say "nothing left to analyse -- done, exiting"
        exit 0
    fi

    if [ "$running" = "True" ]; then
        sleep "$POLL"; continue
    fi

    # Stopped with work outstanding. Resume whether the pause was the budget or
    # an interruption; start-processing resumes an existing run rather than
    # starting a second one.
    say "stopped with $remaining urls left (paused_on_budget=$paused): ${note//_/ }"
    reply="$(curl -sS --max-time 60 -X POST "$BASE/api/projects/$PROJECT/start-processing" 2>>"$LOG")"
    say "resume -> $reply"
    sleep "$POLL"
done
