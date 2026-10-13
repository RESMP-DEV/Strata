# Strix host watcher

`strata-watch.service` is installed on the Strix host as a root systemd service.
It is enabled across reboots and restarts after failure. The service samples
every 20 seconds and writes a heartbeat every 60 seconds to
`/var/lib/strata-watch/events-YYYYMMDDTHH.jsonl`; `/var/lib/strata-watch/state.json`
always holds the latest sample. Events are retained for up to 720 hourly files
or 512 MiB.

The watcher records CPU/memory/I/O pressure, NVMe command and rate counters,
network counters, persistent D-state processes, SSH process/service state,
NVIDIA driver procfs, and filtered journald storage, reset, timeout, hang, OOM,
SSH, NVRM/Xid, and unclean-shutdown messages.

It intentionally does not call `nvidia-smi`. During deployment `nvidia-smi`
entered an uninterruptible state and blocked Python's timeout for more than
eight seconds; the main watcher reads `/proc/driver/nvidia` instead.

## Inspection

```sh
systemctl status strata-watch --no-pager
jq . /var/lib/strata-watch/state.json
tail /var/lib/strata-watch/events-*.jsonl | jq .
sudo smartctl --json --all /dev/nvme0n1
sudo nvme smart-log /dev/nvme0n1
sudo nvme error-log /dev/nvme0n1
```

If a large file reports `Input/output error`, do not retry the full-file read or
start a model pass. Preserve watcher state and inspect the kernel journal first.
