# Operating the Optane KV tier (B550)

The B550's operationally named Optane partition carries two different kinds of state: Strata session
files (`slot_save_path`) and the cold KV disk tier (`--kv-tier`). Both are
*derived* state that the engine can rebuild from the model and the prompt, which
is the single most important fact behind every rule here.

This is operational procedure. What the tier is and what it does not do lives in
`architecture.md`; the measurements live in `evidence/`.

## Identify the device every time

Do not trust `/dev/nvmeN` across reboots. The B550 has a root NVMe, the designated 750 GB device,
and a 21.8 TB HDD, and enumeration order is not stable.

```bash
lsblk -o NAME,PATH,MODEL,SERIAL,SIZE,FSTYPE,MOUNTPOINTS   # expect ET000750KWJTF / PHKS826300DP750BGN
sudo -n smartctl -H -A /dev/nvme1n1                          # PASSED, 0% used, 0 media/data-integrity errors
```

If the model or serial is not the designated device, stop. Everything below is written
against `/dev/nvme1n1p2` only because that is what this host enumerates today.

## Mount for the workload, not for convenience

A cold KV tier is a long-lived service, so the mount is long-lived too. The
defaults that make a data disk convenient are wrong for it.

```bash
sudo -n mount -o uid=$(id -u),gid=$(id -g),noatime /dev/nvme1n1p2 /mnt/optane
```

- `uid=`/`gid=` so the engine writes as itself without a second sudo dance.
- `noatime`, because the tier's whole job is re-reading blocks the kernel has no
  reason to keep. atime updates are a write per read on a device priced for
  random reads.

Do **not** keep `errors=remount-ro`. It is the exFAT default this host mounts
with, and it is actively dangerous here: a filesystem error silently downgrades
the tier to read-only, so the engine keeps answering from a tier that is no
longer being written, and the first sign is a checksum failure rather than a
clear "this disk is read-only" error.

Keep a dedicated, mode-0700 subdirectory per consumer (`strata-kv-tier-*`,
`strata-sessions`). The disaggregated tier treats that directory as its root and
creates one fixed-layout file per QSA layer; the older one-file pager must keep
its own separate directory. The partition also holds unrelated directories; a
task that cleans "its" files must name its own subdirectory, never the mount
point.

## Unmount cleanly, every time, and check afterwards

The kernel logged `exFAT-fs (nvme1n1p2): Volume was not properly unmounted` five
times in one session. exFAT has no journal, so an unclean unmount can lose or
corrupt the directory structure, and the corruption shows up later as data loss
in files that were written correctly.

```bash
sudo -n sync && sudo -n umount /mnt/optane
findmnt -T /mnt/optane | tail -1          # must NOT list /dev/nvme1n1p2
sudo -n dmesg -T | grep -i exfat | tail -3   # must show nothing new
```

`sync` before `umount` is not optional on this host. A driver or service that
kills the process owning the mount produces the same dirty flag, so the only
defence is that every teardown path ends in `sync` then `umount`, and that a
caller which cannot unmount leaves the mount alone rather than remounting over it.

## Check the volume before trusting results

A dirty-flagged exFAT volume still mounts, and reads from it still return data.
That is the failure mode to rule out before a measurement.

```bash
sudo -n fsck.exfat -n /dev/nvme1n1p2     # read-only survey; expect "clean"
```

Run the survey after any unclean unmount and before a cold-tier measurement. On
2026-10-20, after five dirty flags, the survey reported `clean. directories 21,
files 101`, which is what a volume whose data survived looks like. A repair pass
(`fsck.exfat` without `-n`) is an operator action on a user's data and is not part
of an agent run.

## The pager tolerates damage; the filesystem does not

The pager's own contract, and the reason a checksum failure is not a data-loss
incident:

- A group is written as a whole, and its valid bit is set only after the payload
  is on disk.
- A read verifies a CRC over the payload and **refuses** the group on a mismatch,
  counting it, instead of returning the bytes it found.
- A refused group is a performance event, not a wrong answer, because the tier
  is derived state.

That tolerance stops at the filesystem. A corrupt exFAT directory can lose a
file entirely, which no checksum inside the file can detect. The pager protects
block contents; only a clean unmount protects the directory.

## Concurrency

One engine process owns one tier directory. Each per-layer file is opened
read-write, its header adopts the process's random session identity, and a
server request resets the epoch before rewriting its session; two processes on
one directory would interleave those writes. A second engine must use a second
subdirectory.

## Capacity and headroom

The tier is bounded by two independent budgets: the coordinator's L1 group cache
(`--kv-tier-l1`, per QSA layer) and the device (whatever is free on the
partition). The engine's full pinned host allocation remains a separate,
currently unreduced budget.
Track both, because a full device fails writes while a full host cache only
costs throughput.

The aligned I/O group depends on the KV format; it is not universally 32 blocks.
Use the group size from [KV disaggregation](KV_DISAGGREGATION.md).

| KV format | B/cell/layer | B/4-cell block | Blocks/group | Group bytes |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 2,048 | 8,192 | 1 | 8,192 |
| INT8 | 1,056 | 4,224 | 32 | 135,168 |
| Q4_0 | 576 | 2,304 | 16 | 36,864 |
| K8V4 | 1,104 | 4,416 | 64 | 282,624 |

Nominal payload across the 12 QSA layers is:

| Context | FP16 | INT8 | Q4_0 | K8V4 |
| --- | ---: | ---: | ---: | ---: |
| 262,144 cells | 5.81 GiB | 3.09 GiB | 1.55 GiB | 3.15 GiB |
| 600,000 cells | 13.30 GiB | 7.09 GiB | 3.55 GiB | 7.21 GiB |
| 800,000 cells | 17.73 GiB | 9.45 GiB | 4.73 GiB | 9.61 GiB |

These are payload arithmetic, not measured allocations: the engine's own pinned
allocation differs from it, and it did so on this host (3.09 GiB reported for a
context whose payload arithmetic is the same figure but a different measure).

## Health check

```bash
lsblk -o NAME,MODEL,SERIAL,FSTYPE,MOUNTPOINTS /dev/nvme1n1     # mounted as expected
sudo -n smartctl -H -A /dev/nvme1n1 | grep -E "Percentage Used|Media and Data|Temperature"
sudo -n dmesg -T | grep -icE "exFAT.*not properly unmounted"   # should not increase
df -h /mnt/optane
```

`Media and Data Integrity Errors` is the one that matters for this workload; a
nonzero value means the device, not the tier, is the problem.
