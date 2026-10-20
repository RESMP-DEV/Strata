#!/usr/bin/env python3
"""Bounded, receipt-driven B550 CUDA and disaggregated-KV acceptance harness.

The harness reaches B550 only through its configured SSH alias, uses argv lists
rather than a remote shell, verifies the reported host and silicon before use,
and never restarts, signals, or reconfigures an unrelated service.  A gate can
pass only when the engine and its KV-tier report supply the required evidence;
missing evidence is a failed receipt, not a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "build" / "reports" / "b550-acceptance" / "receipt.json"
EXPECTED_GPU = "NVIDIA GeForce RTX 3090 Ti"
EXPECTED_COMPUTE_CAPABILITY = "8.6"
EXPECTED_STORAGE_MODEL = "ET000750KWJTF"
EXPECTED_STORAGE_SERIAL = "PHKS826300DP750BGN"
PINNED_LLAMA_CPP_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
CONTROLLED_ENGINE_FLAGS = {
    "--tokens-file",
    "--max-new",
    "--max-context",
    "--greedy",
    "--seed",
    "--kv-tier",
    "--kv-tier-l1",
    "--serve",
    "--prompt-cache",
    "--batch",
    "--slots",
    "--layer-split",
    "--peer-device",
}
REQUIRED_SERVE_FLAGS = ("--spec", "--prefill")


@dataclass
class Result:
    name: str
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def aggregate_failure(self) -> bool:
        return self.required and not self.passed

    def evidence(self, output_limit: int = 16000) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "exit_code": self.exit_code,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "required": self.required,
            "stdout": _truncate(self.stdout, output_limit),
            "stderr": _truncate(self.stderr, output_limit),
        }


@dataclass
class RunTrace:
    arm: str
    prompt_tokens: list[int]
    output_tokens: list[int]
    nll_values: list[float]
    argmax_values: list[int]
    counters: dict[str, int] | None
    positions: list[int] | None = None
    done: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "prompt_token_count": len(self.prompt_tokens),
            "output_tokens": self.output_tokens,
            "scored_positions": (
                [self.positions[0], self.positions[-1]] if self.positions else None
            ),
            "nll_count": len(self.nll_values),
            "mean_nll": _mean(self.nll_values),
            "perplexity": _perplexity(self.nll_values),
            "counters": self.counters,
            "request": self.done,
        }


@dataclass
class Harness:
    ssh_host: str = "B550"
    ssh_timeout: int = 10
    command_timeout: int = 3600
    runner: Any = None
    steps: list[Result] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def local(
        self, name: str, command: list[str], input: str | None = None, required: bool = True
    ) -> Result:
        return self._execute(name, command, remote=False, input=input, required=required)

    def remote(
        self, name: str, command: list[str], input: str | None = None, required: bool = True
    ) -> Result:
        quoted = " ".join(shlex.quote(part) for part in command)
        wrapped = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.ssh_timeout}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            self.ssh_host,
            quoted,
        ]
        return self._execute(name, wrapped, remote=True, input=input, required=required)

    def _execute(
        self, name: str, command: list[str], remote: bool, input: str | None, required: bool
    ) -> Result:
        started = time.monotonic()
        runner = self.runner or subprocess.run
        try:
            completed = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                input=input,
                timeout=self.command_timeout,
            )
            result = Result(
                name=name,
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                elapsed_seconds=time.monotonic() - started,
                required=required,
            )
        except subprocess.TimeoutExpired as error:
            result = Result(
                name=name,
                command=command,
                exit_code=124,
                stdout=_text(error.stdout),
                stderr=_text(error.stderr) + f"\nharness timeout after {self.command_timeout}s\n",
                elapsed_seconds=time.monotonic() - started,
                required=required,
            )
        except OSError as error:
            result = Result(
                name=name,
                command=command,
                exit_code=127,
                stdout="",
                stderr=f"{type(error).__name__}: {error}",
                elapsed_seconds=time.monotonic() - started,
                required=required,
            )
        self.steps.append(result)
        marker = "REMOTE" if remote else "LOCAL"
        print(
            f"{'PASS' if result.passed else 'FAIL'}{' (optional)' if not result.required else ''} "
            f"{marker} {name}: exit {result.exit_code}",
            flush=True,
        )
        if result.aggregate_failure:
            self.failures.append(
                f"{marker} {name} exited {result.exit_code}: {result.stderr.strip()}"
            )
        return result

    def require(self, condition: bool, failure: str) -> None:
        if not condition and failure not in self.failures:
            self.failures.append(failure)

    def assert_clean(self) -> None:
        if self.failures:
            raise HarnessFailure("; ".join(self.failures))


class HarnessFailure(RuntimeError):
    pass


def _text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 40] + f"\n... truncated {len(value) - limit} bytes ...\n"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _perplexity(values: list[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values)
    assert mean is not None
    try:
        return pow(2.718281828459045, -mean)
    except OverflowError:
        return float("inf")


def _sha256_text(values: list[int]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_gpu_list(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            raise ValueError(f"malformed nvidia-smi row: {line!r}")
        rows.append(
            {
                "name": values[0],
                "uuid": values[1],
                "pci_bus_id": values[2],
                "compute_cap": values[3],
            }
        )
    return rows


def _lsblk_devices(node: dict[str, Any]) -> list[dict[str, Any]]:
    devices = [node]
    for child in node.get("children", []):
        devices.extend(_lsblk_devices(child))
    return devices


def find_optane(text: str) -> dict[str, Any]:
    document = json.loads(text)
    disks = [
        node
        for root in document.get("blockdevices", [])
        for node in _lsblk_devices(root)
        if node.get("type") == "disk"
    ]
    matches = [
        node
        for node in disks
        if node.get("model") == EXPECTED_STORAGE_MODEL
        and node.get("serial") == EXPECTED_STORAGE_SERIAL
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Optane disk, found {len(matches)}")
    return matches[0]


def parse_findmnt(text: str) -> dict[str, Any]:
    document = json.loads(text)
    entries = document.get("filesystems", [])
    if len(entries) != 1:
        raise ValueError(f"expected one findmnt entry, found {len(entries)}")
    entry = entries[0]
    options = [item.strip() for item in str(entry.get("options", "")).split(",")]
    return {
        "source": entry.get("source"),
        "target": entry.get("target"),
        "fstype": entry.get("fstype"),
        "options": options,
        "read_only": "ro" in options,
    }


def optane_mount_error(optane: dict[str, Any], mount: dict[str, Any], target: str) -> str | None:
    if mount.get("target") != target:
        return f"{target} is not a mount point; findmnt resolved {mount.get('target')!r}"
    source = str(mount.get("source", ""))
    disk_path = str(optane.get("path", ""))
    if not source.startswith(disk_path) or re.fullmatch(r"p\d+", source[len(disk_path) :]) is None:
        return f"mount source {source!r} is not a partition of Optane {disk_path!r}"
    return None


def probe_reachability(harness: Harness, address: str, zero_tier_address: str) -> dict[str, Any]:
    """Probe every documented lane and record what answered.

    The B550 is reachable by SSH alias and by a caller-discovered LAN address,
    optionally plus a caller-discovered ZeroTier address.  A failure on one
    lane alone does not say which lane is down, so each requested lane is
    probed and the answer is recorded: reachability is a fact, and no lane
    answering is a blocker, never a pass.
    """
    lanes: dict[str, Any] = {}
    for name, target in (("lan", address), ("zero_tier", zero_tier_address)):
        if not target:
            continue
        probe = harness.local(
            f"reachability-{name}",
            ["nc", "-G", "5", "-z", target, "22"],
            required=False,
        )
        lanes[name] = {
            "target": target,
            "reachable": probe.passed,
            "exit_code": probe.exit_code,
        }
    alias = harness.remote("reachability-alias", ["hostname"], required=False)
    lanes["ssh_alias"] = {
        "target": harness.ssh_host,
        "reachable": alias.passed,
        "exit_code": alias.exit_code,
        "error": alias.stderr.strip()[:400],
    }
    reachable = any(lane["reachable"] for lane in lanes.values())
    harness.facts["reachability"] = {"reachable": reachable, "lanes": lanes}
    harness.require(reachable, "B550 answered on no documented lane; the hardware gate is blocked")
    return lanes


def verify_identity(
    harness: Harness, expected_address: str, expected_hostname: str, optane_mount: str
) -> dict[str, Any]:
    ssh_config = harness.local("ssh-alias", ["ssh", "-G", harness.ssh_host])
    config_values: dict[str, str] = {}
    if ssh_config.passed:
        for line in ssh_config.stdout.splitlines():
            key, separator, value = line.partition(" ")
            if separator and key in {"hostname", "user", "port"}:
                config_values[key] = value
        harness.require(
            config_values.get("hostname") == expected_address,
            f"SSH alias resolves to {config_values.get('hostname')!r}, not {expected_address!r}",
        )

    hostname = harness.remote("reported-hostname", ["hostname"])
    harness.require(
        hostname.passed and hostname.stdout.strip() == expected_hostname,
        f"reported hostname is {hostname.stdout.strip()!r}, not {expected_hostname!r}",
    )

    gpu = harness.remote(
        "gpu-identity",
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    gpu_rows: list[dict[str, str]] = []
    if gpu.passed:
        try:
            gpu_rows = parse_gpu_list(gpu.stdout)
        except ValueError as error:
            harness.require(False, str(error))
    harness.require(len(gpu_rows) == 2, f"expected two GPUs, found {len(gpu_rows)}")
    for index, row in enumerate(gpu_rows):
        harness.require(row["name"] == EXPECTED_GPU, f"GPU {index} is {row['name']!r}")
        harness.require(
            row["compute_cap"] == EXPECTED_COMPUTE_CAPABILITY,
            f"GPU {index} compute capability is {row['compute_cap']!r}",
        )

    lsblk = harness.remote(
        "optane-identity",
        [
            "lsblk",
            "-J",
            "-o",
            "NAME,PATH,TYPE,MODEL,SERIAL,SIZE,FSTYPE,MOUNTPOINTS",
        ],
    )
    optane: dict[str, Any] | None = None
    if lsblk.passed:
        try:
            optane = find_optane(lsblk.stdout)
        except (ValueError, json.JSONDecodeError) as error:
            harness.require(False, f"Optane identity: {error}")
    if optane is None:
        harness.require(False, "the expected Optane model/serial was not found live")

    mounted = harness.remote("optane-mount", ["findmnt", "-J", "-T", optane_mount])
    mount: dict[str, Any] | None = None
    if mounted.passed:
        try:
            mount = parse_findmnt(mounted.stdout)
        except (ValueError, json.JSONDecodeError) as error:
            harness.require(False, f"Optane mount: {error}")
    else:
        harness.require(False, f"{optane_mount} is not mounted; identity never mounts storage")
    if optane is not None and mount is not None:
        failure = optane_mount_error(optane, mount, optane_mount)
        harness.require(
            failure is None,
            failure or "invalid Optane mount",
        )

    facts = {
        "ssh_alias": harness.ssh_host,
        "ssh_config": config_values,
        "reported_hostname": hostname.stdout.strip(),
        "gpus": gpu_rows,
        "optane": optane,
        "optane_mount": mount,
    }
    harness.facts.update(facts)
    return facts


def verify_build(
    harness: Harness,
    *,
    source: str,
    build_dir: str,
    nvcc: str,
    parallel: int,
    expected_revision: str,
    ggml_dir: str = "",
    linker_flags: list[str] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Check out by revision, configure only sm_86, and build the engine/device probes."""
    resolved = harness.remote(
        "remote-source", ["git", "-C", source, "rev-parse", "--show-toplevel"]
    )
    revision = harness.remote("remote-revision", ["git", "-C", source, "rev-parse", "HEAD"])
    status = harness.remote("remote-status", ["git", "-C", source, "status", "--porcelain"])
    harness.require(
        revision.passed and revision.stdout.strip() == expected_revision,
        "remote source revision differs",
    )
    harness.require(
        resolved.passed and resolved.stdout.strip() == source,
        "remote source is not the requested worktree",
    )
    if not allow_dirty:
        harness.require(
            status.passed and not status.stdout.strip(), "remote source has unrelated dirty state"
        )

    if ggml_dir:
        ggml_revision = harness.remote(
            "llamacpp-revision", ["git", "-C", ggml_dir, "rev-parse", "HEAD"]
        )
        ggml_status = harness.remote(
            "llamacpp-status", ["git", "-C", ggml_dir, "status", "--porcelain"]
        )
        harness.require(
            ggml_revision.passed and ggml_revision.stdout.strip() == PINNED_LLAMA_CPP_REVISION,
            "local llama.cpp source differs from the pinned build revision",
        )
        harness.require(
            ggml_status.passed and not ggml_status.stdout.strip(),
            "local llama.cpp source has unrelated dirty state",
        )

    configure = harness.remote(
        "cuda-sm86-configure",
        [
            "cmake",
            "-S",
            source,
            "-B",
            build_dir,
            "-G",
            "Ninja",
            "-DSTRATA_ENABLE_CUDA=ON",
            "-DSTRATA_BUILD_TESTS=ON",
            "-DSTRATA_PORTABLE=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=86",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            *([f"-DSTRATA_GGML_DIR={ggml_dir}"] if ggml_dir else []),
            *([f"-DCMAKE_EXE_LINKER_FLAGS={' '.join(linker_flags)}"] if linker_flags else []),
        ],
    )
    build = harness.remote(
        "cuda-sm86-build",
        [
            "cmake",
            "--build",
            build_dir,
            "--target",
            "strata",
            "strata-device",
            "--parallel",
            str(parallel),
        ],
    )
    harness.require(configure.passed, "CUDA sm_86 configure failed")
    harness.require(build.passed, "CUDA sm_86 engine build failed")
    require_idle_gpus(harness, "device-identity")
    device_identity = harness.remote(
        "strata-device-identity",
        [f"{build_dir.rstrip('/')}/strata-device", "--selftest"],
    )
    device_list = harness.remote(
        "strata-device-list",
        [f"{build_dir.rstrip('/')}/strata-device", "--list-devices"],
    )
    harness.require(
        device_identity.passed and "strata-device selftest OK" in device_identity.stdout,
        "the built engine's runtime device identity selftest failed",
    )
    harness.require(
        "NVIDIA GeForce RTX 3090 Ti" in device_identity.stdout
        and "compute capability  8.6" in device_identity.stdout,
        "the built engine did not identify an RTX 3090 Ti as compute capability 8.6",
    )
    harness.require(
        device_list.passed
        and "cannot run:" not in device_list.stdout
        and device_list.stdout.count("NVIDIA GeForce RTX 3090 Ti") == 2,
        "the built engine did not enumerate both RTX 3090 Ti cards as runnable",
    )
    facts = {
        "remote_source": source,
        "remote_revision": revision.stdout.strip(),
        "remote_dirty": bool(status.stdout.strip()) if status.passed else None,
        "build_dir": build_dir,
        "ggml_dir": ggml_dir or None,
        "linker_flags": linker_flags or [],
    }
    harness.facts.update(facts)
    return facts


def _remote_quoted(
    harness: Harness, name: str, command: list[str], input: str | None = None
) -> Result:
    wrapped = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={harness.ssh_timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        harness.ssh_host,
        " ".join(shlex.quote(part) for part in command),
    ]
    return harness.local(name, wrapped, input=input)


def require_idle_gpus(harness: Harness, context: str) -> None:
    result = harness.remote(
        f"gpu-idle-{context}",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    harness.require(
        result.passed and not result.stdout.strip(),
        "GPU compute applications are present; refusing to contend",
    )


def parse_engine_output(text: str, arm: str) -> RunTrace:
    prompt_match = re.search(r"^prompt  :(.*)$", text, flags=re.MULTILINE)
    output_match = re.search(r"^output  :(.*)$", text, flags=re.MULTILINE)
    if prompt_match is None or output_match is None:
        raise ValueError(f"{arm}: the engine did not print token IDs")
    prompt_tokens = [int(value) for value in prompt_match.group(1).split()]
    output_tokens = [int(value) for value in output_match.group(1).split()]
    if not output_tokens:
        raise ValueError(f"{arm}: the engine emitted no output tokens")

    nll_values: list[float] = []
    argmax_values: list[int] = []
    expected_position = 0
    for line in text.splitlines():
        columns = line.split("\t")
        if len(columns) < 5:
            continue
        try:
            position = int(columns[0])
            _target = int(columns[1])
            log_probability = float(columns[2])
            argmax = int(columns[3])
            _top_probability = float(columns[4])
        except ValueError:
            continue
        if position != expected_position:
            raise ValueError(f"{arm}: STRATA_LOGPOS positions are not contiguous at {position}")
        nll_values.append(log_probability)
        argmax_values.append(argmax)
        expected_position = position + 1
    if len(nll_values) != len(prompt_tokens) - 1:
        raise ValueError(
            f"{arm}: expected {len(prompt_tokens) - 1} teacher-forced values, "
            f"found {len(nll_values)}"
        )
    return RunTrace(arm, prompt_tokens, output_tokens, nll_values, argmax_values, None)


def parse_tier_report(stderr: str, arm: str) -> dict[str, int]:
    reports = re.findall(r"^strata kv tier \([^)]*\): (.*)$", stderr, flags=re.MULTILINE)
    if not reports:
        raise ValueError(
            f"{arm}: the engine printed no KV-tier report, so the cold-read and eviction "
            "counters are not observable; the engine has no call site for the tier's "
            "report, and a run without them cannot prove a cold read happened"
        )
    if len(reports) != 1:
        raise ValueError(f"{arm}: expected one KV-tier report, found {len(reports)}")
    values = {key: int(value) for key, value in re.findall(r"\b([a-z0-9_]+)=(\d+)\b", reports[0])}
    required = (
        "l1_evictions",
        "l2_reads",
        "l2_read_failures",
        "checksum_errors",
        "selection",
        "readiness",
        "resolve",
        "windows",
        "violations",
        "engine_host_restores",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{arm}: KV-tier report lacks {', '.join(missing)}")
    return values


def _token_identities(
    left: RunTrace, right: RunTrace, gate: str, require_positions: bool = True
) -> list[str]:
    failures: list[str] = []
    if left.prompt_tokens != right.prompt_tokens:
        failures.append(f"{gate}: prompt token IDs differ")
    if left.output_tokens != right.output_tokens:
        failures.append(f"{gate}: output token IDs differ")
    if left.argmax_values != right.argmax_values:
        mismatches = sum(
            a != b for a, b in zip(left.argmax_values, right.argmax_values, strict=False)
        )
        failures.append(f"{gate}: {mismatches} prompt argmax mismatches")
    if len(left.nll_values) != len(right.nll_values):
        failures.append(f"{gate}: NLL sample counts differ")
    if require_positions and (left.positions or []) != (right.positions or []):
        failures.append(f"{gate}: the two arms scored different prompt positions")
    return failures


def compare_traces(
    left: RunTrace,
    right: RunTrace,
    tolerance: float,
    exact: bool = False,
    require_positions: bool = True,
) -> dict[str, Any]:
    gate = "identity gate" if exact else "gate"
    failures = _token_identities(left, right, gate, require_positions)
    left_mean = _mean(left.nll_values)
    right_mean = _mean(right.nll_values)
    deltas = [abs(a - b) for a, b in zip(left.nll_values, right.nll_values, strict=False)]
    max_delta = max(deltas, default=float("inf"))
    mean_delta = abs((right_mean or 0.0) - (left_mean or 0.0))
    if exact and max_delta != 0.0:
        failures.append(f"identity gate: maximum prompt-NLL delta is {max_delta:.9g}, not zero")
    elif not exact and mean_delta > tolerance:
        failures.append(f"gate: mean NLL delta {mean_delta:.9g} exceeds {tolerance:.9g}")
    return {
        "passed": not failures,
        "failures": failures,
        "left": left.metadata(),
        "right": right.metadata(),
        "mean_nll_delta": mean_delta,
        "max_abs_logprob_delta": max_delta,
        "prompt_token_sha256_left": _sha256_text(left.prompt_tokens),
        "prompt_token_sha256_right": _sha256_text(right.prompt_tokens),
        "output_token_sha256_left": _sha256_text(left.output_tokens),
        "output_token_sha256_right": _sha256_text(right.output_tokens),
        "argmax_sha256_left": _sha256_text(left.argmax_values),
        "argmax_sha256_right": _sha256_text(right.argmax_values),
        "left_request": left.done,
        "right_request": right.done,
    }


def _validate_engine_args(args: list[str], kv_resident: int) -> None:
    if kv_resident < 20480:
        raise ValueError("--kv-resident must be at least the engine minimum 20480")
    for flag in CONTROLLED_ENGINE_FLAGS:
        if flag in args:
            raise ValueError(f"the harness owns {flag}; remove it from --engine-arg")
    for required in REQUIRED_SERVE_FLAGS:
        if required not in args:
            raise ValueError(f"serve mode needs {required} in --engine-arg")


def _require_writable_tier_parent(harness: Harness, mount: dict[str, Any], tier_dir: str) -> None:
    target = str(mount.get("target", ""))
    harness.require(
        tier_dir.startswith(target.rstrip("/") + "/") and len(tier_dir) > len(target) + 1,
        f"KV tier path {tier_dir!r} is not inside Optane mount {target!r}",
    )
    harness.require(
        not mount.get("read_only", True), "the Optane mount is read-only; the cold tier cannot run"
    )


def _serve_command(
    *,
    engine: str,
    engine_args: list[str],
    max_context: int,
    logpos_path: str,
    cold_tier: str | None,
    l1_groups: int,
    sm86_perf: str | None,
) -> list[str]:
    """The engine's serve command line, with the harness's own flags last.

    Serve mode is the only path that writes the teacher-forced log-probabilities
    (STRATA_LOGPOS) and the only one that runs the KV tier's serving-path hooks,
    so a live-model NLL control has to drive it.
    """
    command = ["env", f"STRATA_LOGPOS={logpos_path}"]
    if sm86_perf is not None:
        command.append(f"STRATA_SM86_PERF={sm86_perf}")
    command.append(engine)
    command.extend(engine_args)
    command.extend(["--serve", "--max-context", str(max_context)])
    if cold_tier is not None:
        command.extend(["--kv-tier", cold_tier, "--kv-tier-l1", str(l1_groups)])
    return command


def parse_done(line: str) -> dict[str, Any]:
    """The engine's DONE fields used by correctness and scale receipts."""
    fields = line.split()
    if len(fields) < 9:
        raise ValueError("the DONE line lacks the generated/prompt/timing fields")
    result: dict[str, Any] = {
        "generated": int(fields[1]),
        "prompt_tokens": int(fields[2]),
        "prompt_ms": float(fields[3]),
        "decode_ms": float(fields[4]),
        "finish": fields[5],
        "drafts_accepted": int(fields[6]),
        "drafts_offered": int(fields[7]),
        "reused": int(fields[8]),
    }
    if len(fields) >= 15:
        result["prompt_tokens_read"] = int(fields[14])
    for item in fields[9:]:
        key, separator, value = item.partition("=")
        if separator and key in {"chunk", "experts_streamed"}:
            result[key] = int(value)
    return result


def parse_serve_output(text: str, arm: str) -> tuple[list[int], str]:
    """The generated token IDs and the DONE line of a serve run."""
    tokens: list[int] = []
    done = ""
    for line in text.splitlines():
        if line.startswith("T "):
            tokens.append(int(line[2:].strip()))
        elif line.startswith("DONE "):
            done = line
    if not tokens:
        raise ValueError(f"{arm}: the engine emitted no token")
    if not done:
        raise ValueError(f"{arm}: the engine did not finish the request")
    fields = done.split()
    if len(fields) > 5 and fields[5] in {"cancel", "nonfinite"}:
        raise ValueError(f"{arm}: the serve request ended {fields[5]}")
    return tokens, done


def parse_logpos(text: str, arm: str) -> tuple[list[int], list[float], list[int]]:
    """The scored positions, their teacher-forced log-probabilities, and their argmax.

    Only the prompt part that went through the engine's verify windows is
    scored, so the positions are absolute prompt positions and do not have to
    start at zero.  They must be contiguous and strictly increasing: a gap
    means the file is not one run's complete trace.
    """
    positions: list[int] = []
    nll_values: list[float] = []
    argmax_values: list[int] = []
    for line in text.splitlines():
        columns = line.split("\t")
        if len(columns) < 5:
            continue
        try:
            position = int(columns[0])
            log_probability = float(columns[2])
            argmax = int(columns[3])
        except ValueError:
            continue
        if not math.isfinite(log_probability):
            raise ValueError(f"{arm}: a non-finite log-probability at position {position}")
        if positions and position != positions[-1] + 1:
            raise ValueError(f"{arm}: STRATA_LOGPOS positions are not contiguous at {position}")
        positions.append(position)
        nll_values.append(log_probability)
        argmax_values.append(argmax)
    if not nll_values:
        raise ValueError(f"{arm}: the engine wrote no teacher-forced log-probabilities")
    return positions, nll_values, argmax_values


def _run_engine(
    harness: Harness,
    *,
    arm: str,
    engine: str,
    engine_args: list[str],
    tokens: list[int],
    max_new: int,
    max_context: int,
    work_dir: str,
    cold_tier: str | None,
    l1_groups: int,
    sm86_perf: str | None,
) -> RunTrace:
    """Run one arm of the control against a live engine on B550."""
    require_idle_gpus(harness, arm)
    harness.remote(f"work-dir-{arm}", ["mkdir", "-p", work_dir])
    harness.assert_clean()
    temporary = harness.remote(f"logpos-path-{arm}", ["mktemp", work_dir + "/logpos.XXXXXX"])
    harness.assert_clean()
    logpos = temporary.stdout.strip()
    command = _serve_command(
        engine=engine,
        engine_args=engine_args,
        max_context=max_context,
        logpos_path=logpos,
        cold_tier=cold_tier,
        l1_groups=l1_groups,
        sm86_perf=sm86_perf,
    )
    request = harness.remote(f"request-path-{arm}", ["mktemp", work_dir + "/gen.XXXXXX"])
    harness.assert_clean()
    request_path = request.stdout.strip()
    request_body = f"GEN {max_new} " + ",".join(str(token) for token in tokens) + "\n"
    harness.remote(
        f"request-{arm}",
        ["sh", "-c", f"cat > {shlex.quote(request_path)}"],
        input=request_body,
    )
    harness.assert_clean()
    pipeline = (
        f"cat -- {shlex.quote(request_path)} | {' '.join(shlex.quote(part) for part in command)}"
    )
    executed = _remote_quoted(harness, f"engine-{arm}", ["sh", "-c", pipeline])
    logpos_text = Result("unused", [], 0, "", "", 0.0)
    if executed.passed:
        logpos_text = harness.remote(f"logpos-{arm}", ["cat", "--", logpos])
    try:
        output_tokens, done = parse_serve_output(executed.stdout, arm)
        request = parse_done(done)
        positions, nll_values, argmax_values = parse_logpos(logpos_text.stdout, arm)
        counters = parse_tier_report(executed.stderr, arm) if cold_tier is not None else None
    except ValueError as error:
        harness.require(False, str(error))
        raise HarnessFailure(str(error)) from error
    return RunTrace(
        arm,
        list(tokens),
        output_tokens,
        nll_values,
        argmax_values,
        counters,
        positions,
        request,
    )


def _artifact_hashes(
    harness: Harness,
    *,
    engine: str,
    engine_args: list[str],
    tokens_file: str,
    max_new: int,
    max_context: int,
    cold_tier: str | None,
    l1_groups: int,
) -> dict[str, Any]:
    model_args: list[str] = []
    for index, flag in enumerate(engine_args):
        if flag in {"--pack", "--native", "--model"} and index + 1 < len(engine_args):
            model_args.append(engine_args[index + 1])
    files = [engine, *model_args, tokens_file]
    hashed = harness.remote("artifact-hashes", ["sha256sum", *files])
    hashes: dict[str, str] = {}
    if hashed.passed:
        for line in hashed.stdout.splitlines():
            digest, separator, path = line.partition(" ")
            if separator and len(path) >= 1 and path[0] in " *":
                hashes[path[1:]] = digest
    harness.require(set(hashes) == set(files), "could not hash every engine/model/token artifact")
    return {
        "engine": engine,
        "engine_args": engine_args,
        "tokens_file": tokens_file,
        "max_new": max_new,
        "max_context": max_context,
        "cold_tier": cold_tier,
        "kv_tier_l1": l1_groups,
        "sha256": hashes,
    }


def prepare_cold_tier(harness: Harness, mount: dict[str, Any], tier_dir: str) -> Result:
    _require_writable_tier_parent(harness, mount, tier_dir)
    created = harness.remote("create-cold-tier", ["mkdir", "--parents", "--mode", "0700", tier_dir])
    state = harness.remote(
        "cold-tier-state",
        ["stat", "--format=%F|%a|%n", tier_dir],
    )
    harness.require(
        state.passed and state.stdout.strip() == f"directory|700|{tier_dir}",
        f"KV tier path is not a private 0700 directory: {state.stdout.strip()!r}",
    )
    occupied = harness.remote(
        "cold-tier-empty",
        ["find", tier_dir, "-mindepth", "1", "-maxdepth", "1", "-print", "-quit"],
    )
    harness.require(
        occupied.passed and not occupied.stdout.strip(),
        "KV tier path is not empty; a matched control must not reuse another consumer's state",
    )
    return created


def run_identity_gate(
    harness: Harness,
    *,
    engine: str,
    engine_args: list[str],
    tokens_file: str,
    max_new: int,
    max_context: int,
    work_dir: str,
) -> dict[str, Any]:
    _validate_engine_args(engine_args, _kv_resident(engine_args))
    tokens_file_result = harness.remote("tokens-file", ["cat", "--", tokens_file])
    harness.assert_clean()
    try:
        tokens = [int(value) for value in tokens_file_result.stdout.replace(",", " ").split()]
    except ValueError as error:
        raise HarnessFailure(f"malformed token file {tokens_file}: {error}") from error
    harness.require(bool(tokens), "the token file is empty")
    harness.assert_clean()
    off = _run_engine(
        harness,
        arm="sm86-off",
        engine=engine,
        engine_args=engine_args,
        tokens=tokens,
        max_new=max_new,
        max_context=max_context,
        work_dir=work_dir,
        cold_tier=None,
        l1_groups=0,
        sm86_perf="0",
    )
    on = _run_engine(
        harness,
        arm="sm86-on",
        engine=engine,
        engine_args=engine_args,
        tokens=tokens,
        max_new=max_new,
        max_context=max_context,
        work_dir=work_dir,
        cold_tier=None,
        l1_groups=0,
        sm86_perf=None,
    )
    result = compare_traces(off, on, tolerance=0.0, exact=True, require_positions=False)
    result["gate"] = "sm86-identity"
    result["artifacts"] = _artifact_hashes(
        harness,
        engine=engine,
        engine_args=engine_args,
        tokens_file=tokens_file,
        max_new=max_new,
        max_context=max_context,
        cold_tier=None,
        l1_groups=0,
    )
    for failure in result["failures"]:
        harness.require(False, failure)
    return result


def _kv_resident(engine_args: list[str]) -> int:
    try:
        return int(engine_args[engine_args.index("--kv-resident") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("--kv-resident N is required after the model arguments") from error


def run_cold_control(
    harness: Harness,
    *,
    engine: str,
    engine_args: list[str],
    tokens_file: str,
    max_new: int,
    max_context: int,
    work_dir: str,
    tier_dir: str,
    l1_groups: int,
    tolerance: float,
) -> dict[str, Any]:
    _validate_engine_args(engine_args, _kv_resident(engine_args))
    if l1_groups < 1:
        raise ValueError("--kv-tier-l1 must be at least one")
    mount = harness.facts.get("optane_mount")
    if not isinstance(mount, dict):
        raise HarnessFailure("the live Optane mount must be verified before the cold control")
    prepare_cold_tier(harness, mount, tier_dir)
    tokens_file_result = harness.remote("tokens-file", ["cat", "--", tokens_file])
    harness.assert_clean()
    try:
        tokens = [int(value) for value in tokens_file_result.stdout.replace(",", " ").split()]
    except ValueError as error:
        raise HarnessFailure(f"malformed token file {tokens_file}: {error}") from error
    harness.require(bool(tokens), "the token file is empty")
    harness.assert_clean()
    off = _run_engine(
        harness,
        arm="cold-off",
        engine=engine,
        engine_args=engine_args,
        tokens=tokens,
        max_new=max_new,
        max_context=max_context,
        work_dir=work_dir,
        cold_tier=None,
        l1_groups=0,
        sm86_perf=None,
    )
    on = _run_engine(
        harness,
        arm="cold-on",
        engine=engine,
        engine_args=engine_args,
        tokens=tokens,
        max_new=max_new,
        max_context=max_context,
        work_dir=work_dir,
        cold_tier=tier_dir,
        l1_groups=l1_groups,
        sm86_perf=None,
    )
    counters = on.counters or {}
    result = compare_traces(off, on, tolerance=tolerance)
    result["gate"] = "matched-cold-control"
    result["cold_checks"] = {
        "cold_reads_nonzero": counters.get("l2_reads", 0) > 0,
        "l1_evictions_nonzero": counters.get("l1_evictions", 0) > 0,
        "read_failures": counters.get("l2_read_failures", -1),
        "checksum_errors": counters.get("checksum_errors", -1),
        "graph_violations": counters.get("violations", -1),
        "graph_windows": counters.get("windows", 0),
    }
    if counters.get("l2_reads", 0) <= 0:
        result["failures"].append("gate: the engine reported no L2 cold read")
    if counters.get("l1_evictions", 0) <= 0:
        result["failures"].append("gate: the engine reported no L1 eviction")
    if counters.get("l2_read_failures", 1) != 0:
        result["failures"].append(
            "gate: the engine reported a cold-read failure or omitted the counter"
        )
    if counters.get("checksum_errors", 1) != 0:
        result["failures"].append(
            "gate: the engine reported a checksum error or omitted the counter"
        )
    if counters.get("violations", 1) != 0:
        result["failures"].append(
            "gate: the engine reported a graph-split violation or omitted the counter"
        )
    if counters.get("windows", 0) <= 0:
        result["failures"].append("gate: no complete graph-split window was reported")
    if counters.get("engine_host_restores", 0) <= 0:
        result["failures"].append(
            "gate: no cold group reached the engine host copy consumed by kv_stream_resolve"
        )
    result["artifacts"] = _artifact_hashes(
        harness,
        engine=engine,
        engine_args=engine_args,
        tokens_file=tokens_file,
        max_new=max_new,
        max_context=max_context,
        cold_tier=tier_dir,
        l1_groups=l1_groups,
    )
    for failure in result["failures"]:
        harness.require(False, failure)
    return result


def receipt(
    *,
    harness: Harness,
    gates: dict[str, Any],
    started_at: float,
    scope: str,
) -> dict[str, Any]:
    step_failures = [step.name for step in harness.steps if step.aggregate_failure]
    passed = not harness.failures and not step_failures
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "agent": "Codex B550 acceptance harness",
        "platform": platform.platform(),
        "source_revision": _source_revision(),
        "source_dirty": _source_dirty(),
        "scope": scope,
        "passed": passed,
        "failures": harness.failures,
        "failed_steps": step_failures,
        "facts": harness.facts,
        "gates": gates,
        "steps": [step.evidence() for step in harness.steps],
    }


def _source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _source_dirty() -> bool:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "scripts/b550_acceptance.py",
            "scripts/tests/test_b550_acceptance.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _identity_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--address",
        required=True,
        help="address discovered live for this run; never a historical fixed IP",
    )
    parser.add_argument("--hostname", default="b550")
    parser.add_argument("--optane-mount", default="/mnt/optane")
    parser.add_argument(
        "--zero-tier-address",
        default="",
        help="optional address discovered live for this run; omit rather than reusing an old IP",
    )
    return parser


def _build_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _identity_parser(parser)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--revision", default=_source_revision())
    parser.add_argument("--nvcc", default="/usr/bin/nvcc")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--ggml-dir",
        default="",
        help="verified pinned llama.cpp checkout to use instead of FetchContent",
    )
    parser.add_argument(
        "--linker-flag",
        action="append",
        default=[],
        help="repeat a host-specific executable linker flag; the receipt records it",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def _engine_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _build_parser(parser)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument(
        "--engine-arg", action="append", default=[], help="repeat a base engine argument"
    )
    return parser


def _expand_engine_args(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(shlex.split(value))
    return expanded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", default="B550", help="SSH alias (never an IP address)")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--receipt", type=Path, default=REPORT)
    parser.add_argument(
        "--evidence", type=Path, help="write a sanitized tracked receipt only when asked"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    identity = commands.add_parser(
        "identity", help="verify host, GPUs, and Optane without mounting anything"
    )
    _identity_parser(identity)

    build = commands.add_parser("build", help="verify identity and build CUDA sm_86")
    _build_parser(build)

    acceptance = commands.add_parser(
        "acceptance",
        help="run selected live-model gates; missing or unreported evidence fails",
    )
    _engine_parser(acceptance)
    acceptance.add_argument("--tier-dir", required=True)
    acceptance.add_argument("--kv-tier-l1", type=int, default=1)
    acceptance.add_argument("--tolerance", type=float, default=1e-4)
    acceptance.add_argument(
        "--phase",
        action="append",
        choices=("identity", "build", "identity-gate", "cold-control"),
        required=True,
    )
    acceptance.add_argument(
        "--scale", type=int, action="append", default=[], help="add a matched scale gate"
    )
    acceptance.add_argument("--skip-identity-for-scale", action="store_true")
    return parser


def _run_gates(arguments: argparse.Namespace) -> tuple[Harness, dict[str, Any]]:
    harness = Harness(
        ssh_host=arguments.ssh,
        ssh_timeout=arguments.connect_timeout,
        command_timeout=arguments.timeout,
    )
    gates: dict[str, Any] = {}
    try:
        probe_reachability(harness, arguments.address, arguments.zero_tier_address)
        verify_identity(harness, arguments.address, arguments.hostname, arguments.optane_mount)
        if (
            "build" in arguments.phase
            or "identity-gate" in arguments.phase
            or "cold-control" in arguments.phase
        ):
            verify_build(
                harness,
                source=arguments.remote_repo,
                build_dir=arguments.build_dir,
                nvcc=arguments.nvcc,
                parallel=arguments.parallel,
                expected_revision=arguments.revision,
                ggml_dir=arguments.ggml_dir,
                linker_flags=arguments.linker_flag,
                allow_dirty=arguments.allow_dirty,
            )
            harness.assert_clean()
        if "identity-gate" in arguments.phase:
            gates["sm86_identity"] = run_identity_gate(
                harness,
                engine=arguments.engine,
                engine_args=_expand_engine_args(arguments.engine_arg),
                tokens_file=arguments.tokens_file,
                max_new=arguments.max_new,
                max_context=arguments.context,
                work_dir=arguments.work_dir,
            )
            harness.assert_clean()
        if "cold-control" in arguments.phase:
            gates["matched_cold_control"] = run_cold_control(
                harness,
                engine=arguments.engine,
                engine_args=_expand_engine_args(arguments.engine_arg),
                tokens_file=arguments.tokens_file,
                max_new=arguments.max_new,
                max_context=arguments.context,
                work_dir=arguments.work_dir,
                tier_dir=arguments.tier_dir,
                l1_groups=arguments.kv_tier_l1,
                tolerance=arguments.tolerance,
            )
            harness.assert_clean()
        for scale in arguments.scale:
            label = f"scale_{scale}"
            if scale not in {600_000, 800_000}:
                raise ValueError(f"--scale accepts only 600000 or 800000, not {scale}")
            if not arguments.skip_identity_for_scale:
                verify_identity(
                    harness, arguments.address, arguments.hostname, arguments.optane_mount
                )
            gates[label] = run_cold_control(
                harness,
                engine=arguments.engine,
                engine_args=_expand_engine_args(arguments.engine_arg),
                tokens_file=arguments.tokens_file,
                max_new=arguments.max_new,
                max_context=scale,
                work_dir=arguments.work_dir,
                tier_dir=f"{arguments.tier_dir.rstrip('/')}-scale-{scale}",
                l1_groups=arguments.kv_tier_l1,
                tolerance=arguments.tolerance,
            )
            harness.assert_clean()
    except (HarnessFailure, ValueError) as error:
        harness.failures.append(str(error))
    return harness, gates


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    started = time.monotonic()
    scope = (
        "Live B550 identity/build/KV acceptance. It does not diagnose unreachable hardware, "
        "prove a host absent from the receipt, or accept missing engine counters."
    )
    harness = Harness(
        ssh_host=arguments.ssh,
        ssh_timeout=arguments.connect_timeout,
        command_timeout=arguments.timeout,
    )
    gates: dict[str, Any] = {}
    try:
        if arguments.command == "identity":
            probe_reachability(harness, arguments.address, arguments.zero_tier_address)
            verify_identity(harness, arguments.address, arguments.hostname, arguments.optane_mount)
        elif arguments.command == "build":
            probe_reachability(harness, arguments.address, arguments.zero_tier_address)
            verify_identity(harness, arguments.address, arguments.hostname, arguments.optane_mount)
            verify_build(
                harness,
                source=arguments.remote_repo,
                build_dir=arguments.build_dir,
                nvcc=arguments.nvcc,
                parallel=arguments.parallel,
                expected_revision=arguments.revision,
                ggml_dir=arguments.ggml_dir,
                linker_flags=arguments.linker_flag,
                allow_dirty=arguments.allow_dirty,
            )
        elif arguments.command == "acceptance":
            harness, gates = _run_gates(arguments)
            scope = (
                "Runs only selected live B550 gates. It never claims unselected phases, does not "
                "infer throughput from correctness, and refuses a cold arm without counters/proof."
            )
        else:  # argparse makes this unreachable.
            raise HarnessFailure("unknown command")
    except (HarnessFailure, ValueError) as error:
        harness.failures.append(str(error))

    harness.failures.extend(
        step.name
        for step in harness.steps
        if step.aggregate_failure and step.name not in harness.failures
    )
    report = receipt(harness=harness, gates=gates, started_at=started, scope=scope)
    report["passed"] = not harness.failures
    write_json(arguments.receipt, report)
    if arguments.evidence is not None:
        durable = dict(report)
        durable["scope"] += (
            " Exact output is bounded; no secret, model payload, or KV bytes are copied."
        )
        write_json(arguments.evidence, durable)
    print(f"b550 acceptance: {'PASS' if report['passed'] else 'FAIL'} receipt={arguments.receipt}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
