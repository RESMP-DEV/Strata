#!/usr/bin/env bash
# Run checkpointed direct reads over a file without relying on an SSH session.
# Each chunk gets a bounded timeout, SHA-256, and durable progress record.

set -uo pipefail

if [ "$#" -ne 4 ]; then
    printf 'usage: %s FILE OUTPUT_DIR CHUNK_MIB CHUNK_TIMEOUT_SECONDS\n' "$0" >&2
    exit 64
fi

file=$1
output_dir=$2
chunk_mib=$3
timeout_seconds=$4

if [ ! -f "$file" ]; then
    printf 'missing file: %s\n' "$file" >&2
    exit 66
fi
if [ "$chunk_mib" -lt 1 ] || [ "$timeout_seconds" -lt 1 ]; then
    printf 'chunk size and timeout must be positive\n' >&2
    exit 64
fi

mkdir -p "$output_dir"
progress=$output_dir/progress.tsv
status=$output_dir/status.tsv
stderr_log=$output_dir/stderr.log
: > "$progress"

file_bytes=$(stat -c '%s' "$file")
chunk_bytes=$((chunk_mib * 1024 * 1024))
file_mib=$(( (file_bytes + 1024 * 1024 - 1) / (1024 * 1024) ))
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'state\tstarted\tfile\tfile_bytes\tchunk_mib\nrunning\t%s\t%s\t%s\t%s\n' \
    "$started" "$file" "$file_bytes" "$chunk_mib" > "$status"

printf 'offset_mib\tbytes\telapsed_seconds\trc\tsha256\n' > "$progress"

pass=1
while [ "$pass" -le 2 ]; do
    offset_mib=0
    while [ "$offset_mib" -lt "$file_mib" ]; do
        if [ $((offset_mib + chunk_mib)) -gt "$file_mib" ]; then
            remaining_bytes=$((file_bytes - offset_mib * 1024 * 1024))
            bs=512
            count_blocks=$((remaining_bytes / 512))
            skip_blocks=$((offset_mib * 2048))
            expected_bytes=$remaining_bytes
        else
            bs=1M
            count_blocks=$chunk_mib
            skip_blocks=$offset_mib
            expected_bytes=$chunk_bytes
        fi
        started_ns=$(date +%s%N)
        raw=$(timeout "$timeout_seconds" \
            dd if="$file" iflag=direct bs="$bs" count="$count_blocks" skip="$skip_blocks" status=none 2>>"$stderr_log" | sha256sum)
        rc=$?
        ended_ns=$(date +%s%N)
        elapsed=$(( (ended_ns - started_ns) / 1000000000 ))
        hash=${raw%% *}
        if [ "$rc" -ne 0 ]; then
            hash=-
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$offset_mib" "$expected_bytes" "$elapsed" "$rc" "$hash" >> "$progress"
        printf 'state\tstarted\tfile\tfile_bytes\tchunk_mib\tpass\toffset_mib\tread_bytes\tlast_rc\tupdated\nfailed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$started" "$file" "$file_bytes" "$chunk_mib" "$pass" "$offset_mib" \
            "$((offset_mib * 1024 * 1024))" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
        if [ "$rc" -ne 0 ]; then
            printf 'direct read failed at pass %s offset %s MiB with exit %s\n' "$pass" "$offset_mib" "$rc" >> "$stderr_log"
            exit "$rc"
        fi
        printf 'state\tstarted\tfile\tfile_bytes\tchunk_mib\tpass\toffset_mib\tread_bytes\tlast_rc\tupdated\nrunning\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$started" "$file" "$file_bytes" "$chunk_mib" "$pass" "$offset_mib" \
            "$((offset_mib * 1024 * 1024))" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
        offset_mib=$((offset_mib + chunk_mib))
    done
    pass=$((pass + 1))
done

printf 'state\tstarted\tfile\tfile_bytes\tchunk_mib\tpass\toffset_mib\tread_bytes\tlast_rc\tupdated\npassed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started" "$file" "$file_bytes" "$chunk_mib" "$((pass - 1))" "$file_mib" "$file_bytes" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
