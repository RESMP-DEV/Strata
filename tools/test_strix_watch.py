#!/usr/bin/env python3
"""Host-safe tests for the persistent Strix watcher."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strix_watch", ROOT / "tools" / "strix_watch.py")
SW = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = SW
SPEC.loader.exec_module(SW)


class StrixWatchTest(unittest.TestCase):
    def test_disk_and_network_rates_exclude_partitions_and_loop_devices(self) -> None:
        disk_text = (
            " 259 0 nvme0n1 100 0 2000 100 200 0 4000 200 3 400 500 0\n"
            " 259 1 nvme0n1p2 100 0 2000 100 200 0 4000 200 3 400 500 0\n"
            " 7 0 loop0 1 0 1 1 0 0 0 0 0 0 0 0\n"
        )
        net_text = (
            "Inter-| Receive | Transmit\n"
            " face |bytes packets errs drop|bytes packets errs drop\n"
            " enp130s0: 1000 10 0 0 0 0 0 0 2000 20 0 0 0 0 0 0\n"
            " ztyjkdkgoy: 500 5 0 0 0 0 0 0 1000 10 0 0 0 0 0 0\n"
        )
        disk_calls = 0
        net_calls = 0

        def fake_read(path: Path) -> str:
            nonlocal disk_calls, net_calls
            if path == Path("/proc/diskstats"):
                disk_calls += 1
                if disk_calls == 2:
                    return (
                        " 259 0 nvme0n1 110 0 2200 110 220 0 4400 220 3 440 550 0\n"
                        " 259 1 nvme0n1p2 110 0 2200 110 220 0 4400 220 3 440 550 0\n"
                    )
                return disk_text
            if path == Path("/proc/net/dev"):
                net_calls += 1
                if net_calls == 2:
                    return (
                        "Inter-| Receive | Transmit\n"
                        " face |bytes packets errs drop|bytes packets errs drop\n"
                        " enp130s0: 1100 11 0 0 0 0 0 0 2200 22 0 0 0 0 0 0\n"
                        " ztyjkdkgoy: 550 6 0 0 0 0 0 0 1100 11 0 0 0 0 0 0\n"
                    )
                return net_text
            return ""

        with (
            mock.patch.object(SW, "read_text", side_effect=fake_read),
            mock.patch.object(SW.time, "monotonic", side_effect=[20.0, 30.0]),
        ):
            disk, _ = SW.parse_diskstats(None)
            disk["_monotonic"] = 10.0
            disk, disk_rates = SW.parse_diskstats(disk)
            network, _ = SW.parse_netdev(None)
            network["_monotonic"] = 20.0
            network, network_rates = SW.parse_netdev(network)
        self.assertEqual(list(disk), ["nvme0n1"])
        self.assertEqual(disk_rates["nvme0n1"]["reads"], 1)
        self.assertEqual(disk_rates["nvme0n1"]["writes"], 2)
        self.assertEqual(set(network_rates), {"enp130s0", "ztyjkdkgoy"})
        self.assertEqual(network_rates["enp130s0"]["rx_bytes"], 10)

    def test_journal_filter_keeps_io_ssh_and_gpu_events(self) -> None:
        output = "\n".join(
            json.dumps(entry)
            for entry in (
                {"__CURSOR": "ordinary", "MESSAGE": "ordinary message"},
                {"__CURSOR": "io", "MESSAGE": "blk_update_request: I/O error", "PRIORITY": "3"},
                {"__CURSOR": "ssh", "MESSAGE": "sshd-session: Connection reset by peer", "PRIORITY": "4"},
                {"__CURSOR": "gpu", "MESSAGE": "NVRM: Xid 79", "PRIORITY": "2"},
            )
        )
        completed = subprocess.CompletedProcess(["journalctl"], 0, stdout=output, stderr="")
        with (
            mock.patch.object(SW.subprocess, "run", return_value=completed),
            mock.patch.object(SW, "current_journal_cursor", return_value="ignored"),
        ):
            matches, cursor, error = SW.drain_journal("old")
        self.assertIsNone(error)
        self.assertEqual(cursor, "gpu")
        self.assertEqual([entry["cursor"] for entry in matches], ["io", "ssh", "gpu"])

    def test_event_sink_writes_jsonl_and_applies_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sink = SW.EventSink(root, retain_hours=1)
            sink.write({"time_epoch": 1_700_000_000.0, "value": 1})
            (root / "events-19000101T00.jsonl").write_text("old\n", encoding="utf-8")
            sink.write({"time_epoch": 1_700_000_001.0, "value": 2})
            paths = sorted(root.glob("events-*.jsonl"))
            self.assertEqual(len(paths), 1)
            events = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["value"] for event in events], [1, 2])

    def test_process_state_parses_parenthesized_comm_and_state(self) -> None:
        process = Path("/proc/8753")
        stat = "8753 (fstrim) D 1 1 1 0 -1..."
        with (
            mock.patch.object(SW.Path, "glob", return_value=[process]),
            mock.patch.object(SW, "read_text", side_effect=[stat, ""]),
        ):
            result = SW.process_states()
        self.assertEqual(result["process_states"]["D"], 1)
        self.assertEqual(result["d_state"], [{"pid": 8753, "comm": "fstrim"}])

    def test_service_is_persistent_and_never_invokes_nvidia_smi(self) -> None:
        unit = (ROOT / "tools" / "strix-watch.service").read_text(encoding="utf-8")
        self.assertIn("Restart=always", unit)
        self.assertIn("WantedBy=multi-user.target", unit)
        self.assertIn("KillMode=process", unit)
        self.assertNotIn("nvidia-smi", unit)


if __name__ == "__main__":
    unittest.main()
