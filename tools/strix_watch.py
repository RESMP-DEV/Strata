#!/usr/bin/env python3
"""Persistent Strix host watcher for I/O, SSH, GPU, and hang evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
except OSError:
    BOOT_ID = "unknown"
DISK_NAME = re.compile(r"^(?:nvme\d+n\d+|sd[a-z]+|vd[a-z]+|mmcblk\d+)$")
JOURNAL_MATCH = re.compile(
    r"(?:i/o error|io error|buffer i/o error|blk_update_request|print_req_error|"
    r"critical medium error|ext4[- ]fs error|jbd2.*error|filesystem read-only|"
    r"read-only file system|nvme.*(?:timeout|reset|abort|error)|ata\d+.*error|"
    r"sd [a-z]+.*error|hung_task|task blocked for more than|kernel panic|"
    r"soft lockup|hard lockup|watchdog.*timeout|rcu.*stall|out of memory|"
    r"oom-killer|systemd-journald.*corrupt|corrupted or uncleanly shut down|"
    r"nvidia.*nvrm|nvidia.*xid|nvrm.*xid|gpu has fallen|driver/library version mismatch|"
    r"nvidia-smi.*failed|sshd?.*(?:error|too many|reset|broken)|"
    r"connection reset by peer|broken pipe|networkd-wait-online.*timeout)",
    re.IGNORECASE,
)
SERVICE_UNITS = (
    "ssh.socket",
    "ssh.service",
    "systemd-journald.service",
    "systemd-networkd.service",
    "NetworkManager.service",
    "zerotier-one.service",
)


@dataclass
class Config:
    state_dir: Path
    interval: float
    heartbeat: float
    once: bool = False


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def parse_key_values(text: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        try:
            output[key] = int(value.strip())
        except ValueError:
            pass
    return output


def parse_labels(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            output[key.strip()] = value.strip()
    return output


def parse_pressure() -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for path in Path("/proc/pressure").glob("*"):
        values: dict[str, float] = {}
        for match in re.finditer(r"(some|full) avg10=([0-9.]+) avg60=([0-9.]+) avg300=([0-9.]+) total=([0-9]+)", read_text(path)):
            scope = match.group(1)
            values.update(
                {
                    f"{scope}_avg10": float(match.group(2)),
                    f"{scope}_avg60": float(match.group(3)),
                    f"{scope}_avg300": float(match.group(4)),
                    f"{scope}_total_us": int(match.group(5)),
                }
            )
        output[path.name] = values
    return output


def parse_meminfo() -> dict[str, int]:
    values = parse_key_values(read_text(Path("/proc/meminfo")))
    keys = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Dirty", "Writeback", "SwapTotal", "SwapFree")
    return {key: values[key] for key in keys if key in values}


def parse_diskstats(previous: dict[str, dict[str, int]] | None = None) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    current: dict[str, dict[str, int]] = {}
    for line in read_text(Path("/proc/diskstats")).splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        if not DISK_NAME.fullmatch(name):
            continue
        current[name] = {
            "reads": int(fields[3]),
            "read_sectors": int(fields[5]),
            "read_ms": int(fields[6]),
            "writes": int(fields[7]),
            "write_sectors": int(fields[9]),
            "write_ms": int(fields[10]),
            "inflight": int(fields[11]),
            "io_ms": int(fields[12]),
            "weighted_io_ms": int(fields[13]),
        }
    rates: dict[str, dict[str, int]] = {}
    if previous:
        now = time.monotonic()
        elapsed = max(now - previous.get("_monotonic", now), 0.001)
        for name, now in current.items():
            before = previous.get(name)
            if before is None:
                continue
            rates[name] = {
                key: max(0, round((now[key] - before[key]) / elapsed))
                for key in now
            }
            rates[name]["inflight"] = now["inflight"]
    return current, rates


def parse_netdev(previous: dict[str, dict[str, int]] | None = None) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    current: dict[str, dict[str, int]] = {}
    lines = read_text(Path("/proc/net/dev")).splitlines()
    for line in lines[2:]:
        name, _, numbers = line.partition(":")
        name = name.strip()
        fields = [int(value) for value in numbers.split()]
        if len(fields) < 16:
            continue
        current[name] = {
            "rx_bytes": fields[0],
            "rx_packets": fields[1],
            "rx_errors": fields[2],
            "rx_dropped": fields[3],
            "tx_bytes": fields[8],
            "tx_packets": fields[9],
            "tx_errors": fields[10],
            "tx_dropped": fields[11],
        }
    rates: dict[str, dict[str, int]] = {}
    if previous:
        now = time.monotonic()
        elapsed = max(now - previous.get("_monotonic", now), 0.001)
        for name, now in current.items():
            before = previous.get(name)
            if before is None:
                continue
            rates[name] = {
                key: max(0, round((now[key] - before[key]) / elapsed))
                for key in now
            }
    return current, rates


def process_states() -> dict[str, Any]:
    counts = {"R": 0, "D": 0, "Z": 0, "S": 0}
    d_state: list[dict[str, Any]] = []
    ssh_children: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        stat = read_text(proc / "stat")
        if not stat:
            continue
        matched = re.match(r"^(\d+) \((.*)\) (\w)", stat)
        if matched is None:
            continue
        pid = int(matched.group(1))
        comm = matched.group(2)
        state = matched.group(3)
        counts[state] = counts.get(state, 0) + 1
        cmdline = read_text(proc / "cmdline").replace("\0", " ").strip()
        if state == "D":
            d_state.append({"pid": pid, "comm": comm})
        if comm in {"sshd", "sshd-session"}:
            ssh_children.append({"pid": pid, "state": state, "cmdline": cmdline[:300]})
    return {
        "process_states": counts,
        "d_state": d_state,
        "ssh_processes": ssh_children,
    }


def query_systemd() -> tuple[dict[str, str], str | None]:
    try:
        completed = subprocess.run(
            ["systemctl", "is-active", *SERVICE_UNITS],
            timeout=5,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {unit: "unknown" for unit in SERVICE_UNITS}, str(error)
    states = [line.strip() or "unknown" for line in completed.stdout.splitlines()]
    if len(states) != len(SERVICE_UNITS):
        states += ["unknown"] * (len(SERVICE_UNITS) - len(states))
    return dict(zip(SERVICE_UNITS, states, strict=True)), None if completed.returncode == 0 else f"exit={completed.returncode}"


def query_gpu() -> dict[str, Any]:
    version = read_text(Path("/proc/driver/nvidia/version")).splitlines()
    driver = version[0].strip() if version else ""
    gpus: list[dict[str, str]] = []
    try:
        for gpu in Path("/proc/driver/nvidia/gpus").glob("*"):
            info = parse_labels(read_text(gpu / "information"))
            gpus.append(
                {
                    "pci_bus_id": gpu.name,
                    "model": str(info.get("Model", "")),
                    "bus_location": str(info.get("Bus Location", "")),
                    "device_minor": str(info.get("Device Minor", "")),
                }
            )
    except OSError:
        pass
    return {"ok": bool(driver), "driver_version": driver, "gpus": gpus}


def current_journal_cursor() -> str | None:
    try:
        completed = subprocess.run(
            ["journalctl", "-n", "1", "-o", "json", "--no-pager"],
            timeout=8,
            check=False,
            capture_output=True,
            text=True,
        )
        entries = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        cursor = entries[-1].get("__CURSOR") if entries else None
        return cursor if isinstance(cursor, str) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def drain_journal(cursor: str | None) -> tuple[list[dict[str, Any]], str | None, str | None]:
    command = ["journalctl", "-o", "json", "--no-pager", "--lines=5000"]
    if cursor:
        command += ["--after-cursor", cursor]
    else:
        return [], current_journal_cursor(), None
    try:
        completed = subprocess.run(command, timeout=12, check=False, capture_output=True, text=True)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], cursor, f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        return [], cursor, completed.stderr.strip()[:500] or f"exit={completed.returncode}"
    matches: list[dict[str, Any]] = []
    latest = cursor
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        candidate = entry.get("__CURSOR")
        if isinstance(candidate, str):
            latest = candidate
        message = entry.get("MESSAGE")
        if not isinstance(message, str) or not JOURNAL_MATCH.search(message):
            continue
        matches.append(
            {
                "time": entry.get("__REALTIME_TIMESTAMP"),
                "priority": entry.get("PRIORITY"),
                "unit": entry.get("_SYSTEMD_UNIT") or entry.get("SYSLOG_IDENTIFIER"),
                "message": message[:4000],
                "cursor": candidate,
            }
        )
    return matches, latest, None


class EventSink:
    def __init__(self, state_dir: Path, max_bytes: int = 64 * 1024 * 1024, retain_hours: int = 720):
        self.state_dir = state_dir
        self.max_bytes = max_bytes
        self.retain_hours = retain_hours
        self.path: Path | None = None

    def write(self, event: dict[str, Any]) -> None:
        stamp = time.gmtime(event["time_epoch"])
        path = self.state_dir / f"events-{stamp.tm_year:04d}{stamp.tm_mon:02d}{stamp.tm_mday:02d}T{stamp.tm_hour:02d}.jsonl"
        if path != self.path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = path
        if path.exists() and path.stat().st_size >= self.max_bytes:
            rotated = path.with_suffix(f".{time.time_ns()}.jsonl")
            path.replace(rotated)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        self.rotate()

    def rotate(self) -> None:
        paths = sorted(self.state_dir.glob("events-*.jsonl"))
        for path in paths[: max(0, len(paths) - self.retain_hours)]:
            path.unlink(missing_ok=True)
        while sum(path.stat().st_size for path in self.state_dir.glob("events-*.jsonl")) > 512 * 1024 * 1024:
            oldest = next(iter(sorted(self.state_dir.glob("events-*.jsonl"))), None)
            if oldest is None:
                break
            oldest.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": event_type,
        "time_epoch": time.time(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic": time.monotonic(),
        "boot_id": BOOT_ID,
        "data": data,
    }


def sample(config: Config, sink: EventSink, state: dict[str, Any]) -> None:
    now_epoch = time.time()
    now_monotonic = time.monotonic()
    previous_disk = dict(state.get("disk", {}))
    previous_disk["_monotonic"] = state.get("monotonic", now_monotonic)
    previous_net = dict(state.get("network", {}))
    previous_net["_monotonic"] = state.get("monotonic", now_monotonic)
    disk, disk_rates = parse_diskstats(previous_disk)
    network, network_rates = parse_netdev(previous_net)
    disk["_monotonic"] = now_monotonic
    network["_monotonic"] = now_monotonic
    proc = process_states()
    services, service_error = query_systemd()
    gpu = query_gpu()
    pressure = parse_pressure()
    journal_events, cursor, journal_error = drain_journal(state.get("journal_cursor"))

    load_parts = read_text(Path("/proc/loadavg")).split()
    load = [float(value) for value in load_parts[:3]] if len(load_parts) >= 3 else []
    metric = {
        "load": load,
        "memory": parse_meminfo(),
        "pressure": pressure,
        "disk": {name: values for name, values in disk.items() if name != "_monotonic"},
        "disk_rates": disk_rates,
        "network": {name: values for name, values in network.items() if name != "_monotonic"},
        "network_rates": network_rates,
        "processes": proc,
        "services": services,
        "gpu": gpu,
        "journal_events": len(journal_events),
    }

    prior_d = {entry["pid"] for entry in state.get("d_state", [])}
    current_d = {entry["pid"] for entry in proc["d_state"]}
    persistent_d = prior_d & current_d
    if persistent_d:
        sink.write(
            make_event(
                "d-state-persisted",
                {"pids": sorted(persistent_d), "processes": proc["d_state"]},
            )
        )
    previous_services = state.get("services", {})
    changed_services = {unit: {"from": previous_services.get(unit), "to": state_value} for unit, state_value in services.items() if previous_services.get(unit) != state_value}
    if changed_services:
        sink.write(make_event("service-state-change", changed_services))
    previous_gpu = state.get("gpu", {})
    if gpu.get("ok") != previous_gpu.get("ok"):
        sink.write(make_event("gpu-state-change", {"from": previous_gpu, "to": gpu}))
    io_pressure = pressure.get("io", {})
    if io_pressure.get("full_avg10", 0) >= 50 or io_pressure.get("some_avg10", 0) >= 80:
        if now_epoch - state.get("last_io_pressure_event", 0) >= 300:
            sink.write(make_event("io-pressure", io_pressure))
            state["last_io_pressure_event"] = now_epoch
    for name, values in disk.items():
        if name != "_monotonic" and values.get("inflight", 0) >= 128:
            sink.write(make_event("disk-high-inflight", {"device": name, "stats": values, "rates": disk_rates.get(name, {})}))
    for entry in journal_events:
        sink.write(make_event("journal-match", entry))
    if journal_error and journal_error != state.get("journal_error"):
        sink.write(make_event("journal-drain-error", {"error": journal_error}))
    if service_error and service_error != state.get("service_error"):
        sink.write(make_event("systemd-query-error", {"error": service_error}))

    state.update(
        boot_id=BOOT_ID,
        updated_epoch=now_epoch,
        monotonic=now_monotonic,
        disk=disk,
        network=network,
        d_state=proc["d_state"],
        services=services,
        gpu=gpu,
        journal_cursor=cursor,
        journal_error=journal_error,
        service_error=service_error,
        last_metric=metric,
    )
    atomic_json(config.state_dir / "state.json", state)
    if now_epoch - state.get("last_heartbeat", 0) >= config.heartbeat:
        sink.write(make_event("heartbeat", metric))
        state["last_heartbeat"] = now_epoch
        atomic_json(config.state_dir / "state.json", state)


def run(config: Config) -> int:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = config.state_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.setdefault("started_epoch", time.time())
    state.setdefault("last_heartbeat", 0)
    sink = EventSink(config.state_dir)
    sink.write(make_event("watcher-start", {"interval": config.interval, "heartbeat": config.heartbeat, "pid": os.getpid()}))
    stopping = False

    def stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        sample(config, sink, state)
        if config.once:
            break
        time.sleep(config.interval)
    if not config.once:
        sink.write(make_event("watcher-stop", {"reason": "signal"}))
    state["stopped_epoch"] = time.time()
    atomic_json(state_path, state)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/strata-watch"))
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--heartbeat", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.interval <= 0 or arguments.heartbeat <= 0:
        parser.error("interval and heartbeat must be positive")
    return run(Config(arguments.state_dir, arguments.interval, arguments.heartbeat, arguments.once))


if __name__ == "__main__":
    raise SystemExit(main())
