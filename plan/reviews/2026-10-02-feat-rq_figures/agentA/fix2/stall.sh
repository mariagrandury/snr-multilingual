PROGRESS_FILE=$1; CELL_ITER=$2; CELL_ACTION=resume; CHAIN_LOGDIR=$(dirname $1)
	stalls=0; progress_recorded=
	if [ -n "$CELL_ACTION" ] && [ -d "$CHAIN_LOGDIR" ]; then
		read -r prev_iter prev_stalls 2>/dev/null < "$PROGRESS_FILE" || prev_iter=
		case ${prev_stalls:-} in ''|*[!0-9]*) prev_stalls=0 ;; esac
		if [ "$CELL_ITER" = "$prev_iter" ]; then stalls=$((prev_stalls + 1)); fi
		echo "$CELL_ITER $stalls" > "$PROGRESS_FILE" 2>/dev/null && progress_recorded=1
	fi
if [ "$stalls" -ge "${CHAIN_MAX_STALLS:-4}" ]; then echo "link@$CELL_ITER stalls=$stalls -> NO successor"; else echo "link@$CELL_ITER stalls=$stalls -> successor queued"; fi
