#!/usr/bin/env bash
# Checkpointed direct read of one bounded logical range. Safe for a block device
# when invoked read-only, but requires caller authorization and root privileges.

set -uo pipefail

if [ "$#" -ne 6 ]; then
    printf 'usage: %s INPUT OUTPUT_DIR START_MIB LENGTH_MIB CHUNK_MIB TIMEOUT_SECONDS\n' "$0" >&2
    exit 64
fi

input=$1
output_dir=$2
start_mib=$3
length_mib=$4
chunk_mib=$5
timeout_seconds=$6

if [ "$start_mib" -lt 0 ]; then
    printf 'start offset must be nonnegative\n' >&2
    exit 64
fi
for value in "$length_mib" "$chunk_mib" "$timeout_seconds"; do
    if [ "$value" -lt 1 ]; then
    printf 'length, chunk, and timeout must be positive\n' >&2
        exit 64
    fi
done
if [ ! -r "$input" ]; then
    printf 'missing or unreadable input: %s\n' "$input" >&2
    exit 66
fi

mkdir -p "$output_dir"
progress=$output_dir/progress.tsv
status=$output_dir/status.tsv
stderr_log=$output_dir/stderr.log
: > "$progress"
: > "$stderr_log"

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'state\tstarted\tinput\tstart_mib\tlength_mib\tchunk_mib\nrunning\t%s\t%s\t%s\t%s\t%s\n' \
    "$started" "$input" "$start_mib" "$length_mib" "$chunk_mib" > "$status"
printf 'offset_mib\tbytes\telapsed_seconds\trc\tsha256\n' > "$progress"

end_mib=$((start_mib + length_mib))
offset_mib=$start_mib
while [ "$offset_mib" -lt "$end_mib" ]; do
    if [ $((offset_mib + chunk_mib)) -gt "$end_mib" ]; then
        count_mib=$((end_mib - offset_mib))
    else
        count_mib=$chunk_mib
    fi
    bytes=$((count_mib * 1024 * 1024))
    started_ns=$(date +%s%N)
    raw=$(timeout "$timeout_seconds" \
        dd if="$input" iflag=direct bs=1M count="$count_mib" skip="$offset_mib" status=none 2>>"$stderr_log" | sha256sum)
    rc=$?
    ended_ns=$(date +%s%N)
    elapsed=$(( (ended_ns - started_ns) / 1000000000 ))
    hash=${raw%% *}
    [ "$rc" -eq 0 ] || hash=-
    printf '%s\t%s\t%s\t%s\t%s\n' "$offset_mib" "$bytes" "$elapsed" "$rc" "$hash" >> "$progress"
    printf 'state\tstarted\tinput\tstart_mib\tlength_mib\tchunk_mib\toffset_mib\tread_bytes\tlast_rc\tupdated\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$([ "$rc" -eq 0 ] && printf running || printf failed)" "$started" "$input" "$start_mib" \
        "$length_mib" "$chunk_mib" "$offset_mib" "$(( (offset_mib - start_mib + count_mib) * 1024 * 1024 ))" \
        "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
    if [ "$rc" -ne 0 ]; then
        printf 'target read failed at offset %s MiB with exit %s\n' "$offset_mib" "$rc" >> "$stderr_log"
        exit "$rc"
    fi
    offset_mib=$((offset_mib + chunk_mib))
done

printf 'state\tstarted\tinput\tstart_mib\tlength_mib\tchunk_mib\toffset_mib\tread_bytes\tlast_rc\tupdated\npassed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started" "$input" "$start_mib" "$length_mib" "$chunk_mib" "$offset_mib" \
    "$((length_mib * 1024 * 1024))" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
