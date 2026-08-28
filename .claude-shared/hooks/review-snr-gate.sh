#!/bin/bash
# review-snr-gate.sh — "review before every commit", enforced by the harness.
#
# The /review-snr skill ends by running `--mark`, which records a fingerprint
# of the pending state (every tracked change vs HEAD, plus the content of
# every untracked file). As a Claude Code PreToolUse hook on Bash, this script
# blocks any `git commit` aimed at this repo whose pending state does not match
# that fingerprint — so a commit is only possible on exactly the tree the
# review saw. Edit anything after the review (including a fix the review
# itself made after marking) and the commit is refused until /review-snr runs
# again. Only Claude's commits go through the hook; your own terminal is
# unaffected.
#
#   review-snr-gate.sh --mark          # written by the skill at the end of a review
#   review-snr-gate.sh --fingerprint   # print the current fingerprint
#   review-snr-gate.sh                 # hook mode: reads the tool call as JSON on stdin
#
# Wired in ~/.claude/settings.json (user scope, because sessions are often
# started from the parent directory, where project settings do not load).
REPO=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
MARK=$REPO/.git/review-snr.ok

fingerprint() {
  cd "$REPO" || exit 0
  { git rev-parse HEAD
    git diff HEAD
    git ls-files --others --exclude-standard -z | xargs -0 -r git hash-object --stdin-paths 2>/dev/null < /dev/null
    git ls-files --others --exclude-standard
  } | sha256sum | cut -c1-16
}

case "${1:-}" in
  --fingerprint) fingerprint; exit 0;;
  --mark) fingerprint > "$MARK" && echo "reviewed state marked: $(cat "$MARK")"; exit 0;;
esac

# Hook mode. Only `git commit` in THIS repo is gated: by cwd, or by the command
# naming the repo (cd/-C). Anything else passes straight through.
input=$(cat)
case "$input" in *commit*) ;; *) exit 0;; esac   # cheap bail-out before parsing
IFS=$'\t' read -r cwd cmd < <(printf '%s' "$input" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d.get("cwd", ""), d.get("tool_input", {}).get("command", "").replace("\n", " "), sep="\t")')
# `git commit` / `git -C <path> commit` as an actual command word — not a
# `commit` that merely appears in a message, a grep, or a log flag.
[[ "$cmd" =~ (^|[\;\&\|[:space:]])git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+commit([[:space:]]|$) ]] || exit 0
case "$cwd $cmd" in *snr-multilingual*) ;; *) exit 0;; esac

want=$(cat "$MARK" 2>/dev/null)
have=$(fingerprint)
[ -n "$want" ] && [ "$want" = "$have" ] && exit 0
echo "review-snr gate: the pending changes in snr-multilingual have not been reviewed" \
     "(marker ${want:-absent}, tree $have). Run /review-snr — it ends by marking the" \
     "reviewed state — then commit exactly that tree." >&2
exit 2
