#!/usr/bin/env python3
"""Run the tiny Qwen4-Exp/RCO compatibility probe on one Modal GPU."""

from __future__ import annotations

import json
from pathlib import Path

import modal

RCO_REVISION = "9a1e09c07d468109cbe60a1b87d5036034a79d10"
LOCAL_PROBE = Path(__file__).with_name("validate_qwen4exp_rco.py")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "accelerate",
        "datasets",
    )
    .run_commands(
        "git clone https://github.com/IST-DASLab/RCO.git /opt/RCO "
        f"&& git -C /opt/RCO checkout {RCO_REVISION}"
    )
    .add_local_file(LOCAL_PROBE, "/root/validate_qwen4exp_rco.py")
)

app = modal.App("strata-qwen4exp-rco-probe", image=image)


@app.function(gpu="L4", timeout=600)
def probe() -> str:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "/root/validate_qwen4exp_rco.py",
            "--rco-root",
            "/opt/RCO",
            "--device",
            "cuda",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    result = json.loads(completed.stdout)
    result["modal_gpu"] = True
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main() -> str:
    result = probe.remote()
    print(result)
    return result
