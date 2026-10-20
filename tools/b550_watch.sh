#!/usr/bin/env bash
# Watch for the B550 and capture the first successful login, its boot window, and the previous session's logs.
#
# The Mac's LAN address changes with the network it joins, so the ZeroTier path is the reliable one. The watcher
# tries every known lane in turn, and on the first success it records the host's uptime, the boot time, the
# shutdown record, the previous boot's kernel and hardware logs, the SMART state of the Optane and the NVMe, and
# the GPU and service state.
set -u

LOG="${B550_WATCH_LOG:-/tmp/b550-watch.log}"
INTERVAL="${B550_WATCH_INTERVAL:-20}"
MAX_TRIES="${B550_WATCH_MAX_TRIES:-1440}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

capture() {
    ssh -o BatchMode=yes -o ConnectTimeout=25 -o ConnectionAttempts=2 "$1" \
         'printf "HOST=%s\nUP=%s\nBOOT=%s\n" "$(hostname)" "$(uptime -p)" "$(uptime -s)";
          echo "--- LAST SHUTDOWN/REBOOT ---"; last -x 2>/dev/null | head -12;
          echo "--- PREVIOUS BOOT KERNEL ERRORS ---";
          journalctl -k -b -1 -p err --no-pager 2>/dev/null | tail -80;
          echo "--- PREVIOUS BOOT TAIL ---";
          journalctl -b -1 --no-pager 2>/dev/null | tail -60;
          echo "--- CURRENT BOOT ERRORS ---";
          journalctl -k -b 0 -p err --no-pager 2>/dev/null | tail -40;
          echo "--- GPU ---"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader;
          echo "--- GPU PROCS ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader;
          echo "--- OPTANE MOUNT ---"; findmnt -no SOURCE,FSTYPE,TARGET,OPTIONS /mnt/optane 2>/dev/null || echo "not mounted";
          echo "--- OPTANE SMART ---"; sudo -n smartctl -A /dev/nvme1n1 2>/dev/null | head -30 || echo "smartctl needs privileges";
          echo "--- STRATA PROCS ---"; pgrep -af "strata|serve/server.py" || echo none;
          echo "--- ZEROTIER ---"; zerotier-cli info 2>/dev/null | head -6 || echo "no zerotier";
          echo "--- FILESYSTEM ---"; df -h / /mnt/optane 2>/dev/null' >"${LOG}.b550" 2>&1
}

log "watch: start; tries=$MAX_TRIES interval=${INTERVAL}s"

for attempt in $(seq 1 "$MAX_TRIES"); do
    # Lane 1: the ZeroTier address.
    if nc -z -G 3 10.243.130.224 22 2>/dev/null; then
        log "watch: the ZeroTier lane answered on attempt $attempt"
        if capture b550-zt; then
            log "watch: the B550 answered and its state was captured in ${LOG}.b550"
            exit 0
        fi
        log "watch: the lane answered but the login failed"
    fi
    # Lane 2: the LAN address, for the times the Mac is on the home network.
    if nc -z -G 2 192.168.1.171 22 2>/dev/null; then
        log "watch: the LAN lane answered on attempt $attempt"
        if capture B550; then
            log "watch: the B550 answered and its state was captured in ${LOG}.b550"
            exit 0
        fi
    fi
    sleep "$INTERVAL"
done

log "watch: gave up after $MAX_TRIES attempts"
exit 1
