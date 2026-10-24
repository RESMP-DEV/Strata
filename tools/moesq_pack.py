"""tools/moesq_pack.py - convert MoESQ paired-sparse NVFP4 experts to Strata NVFP4.

The published Qwen3.8-Flash-Next P48NVFP4-MoESQ checkpoint stores each routed
expert projection as paired-4:8 sparse NVFP4.  Strata's native expert path uses
ggml's dense block_nvfp4 layout, with one E4M3 scale per 16 values and a
16-byte per-expert scale tail.  This tool expands the sparse representation and
repacks it without changing an FP4 code or E4M3 scale byte.  MoESQ's global
scale is a divisor, so the Strata tail stores the nearest F32 reciprocal; that
is one bounded rounding in the effective global multiplier, not bit-exact
effective-weight equality.

The conversion preserves the packed codes and sub-block scales.  It does not preserve MoESQ's
2:4/4:8 structured-sparsity speedup and does not implement MoESQ's activation
global-scale contract: Strata W4A8 uses Q8_1 activations.  Those limits are
recorded in experts.bin.src.json rather than claimed away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

QK = 64
QK_SUB = 16
N_SUB = QK // QK_SUB
NVFP4_TYPE = 40
N_EMBD = 2560
N_FF = 640
ROLES = ("gate_proj", "up_proj", "down_proj")
ROW_SHAPES = {
    "gate_proj": (N_FF, N_EMBD),
    "up_proj": (N_FF, N_EMBD),
    "down_proj": (N_EMBD, N_FF),
}
GGML_ROW_BYTES = 921600
BLOB_BYTES = 3 * GGML_ROW_BYTES + 16
FULL_LAYERS = 48
FULL_EXPERTS = 512
BASE_FILES = ("dense.bin", "index.txt", "conversions.json", "compat-bf16.json")


def read_safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Read a safetensors header without importing safetensors or torch."""
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path.name}: missing safetensors header size")
        size = struct.unpack("<Q", raw_size)[0]
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError(f"{path.name}: truncated safetensors header")
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid safetensors JSON: {exc}") from exc
    return 8 + size, {k: v for k, v in header.items() if k != "__metadata__"}


class Checkpoint:
    """Lazy tensor reader over a MoESQ safetensors snapshot."""

    def __init__(self, root: Path, layers: int = FULL_LAYERS, experts: int = FULL_EXPERTS):
        self.root = root.resolve()
        self.layers = layers
        self.experts = experts
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing {index_path.name} in {self.root}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.headers: dict[Path, tuple[int, dict[str, dict[str, Any]]]] = {}
        self.maps: dict[Path, np.ndarray] = {}
        self.tensor_paths: dict[str, Path] = {}
        self._validate_index()

    def _header(self, path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
        if path not in self.headers:
            self.headers[path] = read_safetensors_header(path)
        return self.headers[path]

    def _map(self, path: Path) -> np.ndarray:
        if path not in self.maps:
            size = path.stat().st_size
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))
        return self.maps[path]

    def _validate_index(self) -> None:
        layer_names = {name.split(".")[3] for name in self.index if ".mlp.experts." in name}
        wanted_layers = {str(i) for i in range(self.layers)}
        if not wanted_layers.issubset(layer_names):
            got = sorted(int(n) for n in layer_names if n.isdigit())
            raise ValueError(
                f"expected leading layers 0..{self.layers - 1}, "
                f"got {got[0] if got else None}..{got[-1] if got else None} ({len(got)} layers)"
            )
        for layer in range(self.layers):
            for expert in range(self.experts):
                for role in ROLES:
                    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}"
                    for suffix in (
                        "weight_sparse_packed",
                        "weight_sparse_mask",
                        "weight_scale",
                        "weight_global_scale",
                    ):
                        if f"{prefix}.{suffix}" not in self.index:
                            raise ValueError(f"missing tensor {prefix}.{suffix}")
        for name, shard in self.index.items():
            path = self.root / shard
            if not path.is_file():
                raise FileNotFoundError(f"{name}: missing shard {path}")
            self.tensor_paths[name] = path.resolve()

    def tensor(self, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        if name not in self.index:
            raise KeyError(name)
        path = self.tensor_paths[name]
        data_start, header = self._header(path)
        if name not in header:
            raise ValueError(f"{path.name}: index names {name}, but the shard does not contain it")
        info = header[name]
        begin, end = info["data_offsets"]
        data = self._map(path)[data_start + begin : data_start + end]
        return data, info

    def scalar_f32(self, name: str) -> float:
        data, info = self.tensor(name)
        if info.get("dtype") != "F32" or info.get("shape") not in ([1], []):
            raise ValueError(f"{name}: expected a scalar F32 tensor, got {info}")
        return float(np.frombuffer(data, dtype="<f4")[0])


def decompress_pair_bitmask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """MoESQ paired-4:8 storage -> dense compressed-tensors packed NVFP4.

    Each mask nibble marks exactly two of four packed bytes.  The first kept
    byte is v0 and the second v1; the selective-where chain below restores each
    byte plane without materializing indices.
    """
    if values.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("paired sparse tensors must be U8")
    rows, values_last = values.shape
    mask_last = mask.shape[-1]
    if mask.shape != (rows, mask_last) or values_last != 4 * mask_last:
        raise ValueError(f"sparse shapes {values.shape} and {mask.shape} are inconsistent")

    nib = np.stack((mask & 0xF, mask >> 4), axis=-1).reshape(rows, mask_last * 2)
    keep = [((nib >> bit) & 1).astype(bool) for bit in range(4)]
    popcount = sum(bit.astype(np.uint8) for bit in keep)
    if int(np.count_nonzero(popcount != 2)):
        raise ValueError(
            f"{int(np.count_nonzero(popcount != 2))} invalid paired-sparse mask nibble(s)"
        )

    pairs = values.reshape(rows, mask_last * 2, 2)
    v0, v1 = pairs[..., 0], pairs[..., 1]
    planes = (
        np.where(keep[0], v0, 0),
        np.where(keep[1], np.where(keep[0], v1, v0), 0),
        np.where(keep[2], np.where(keep[0] | keep[1], v1, v0), 0),
        np.where(keep[3], v1, 0),
    )
    return np.stack(planes, axis=-1).reshape(rows, mask_last * 8)


def expand_scale_32_to_16(scale: np.ndarray, values_per_row: int) -> np.ndarray:
    """Duplicate MoESQ's one scale per 32 values into two 16-value slots."""
    rows, groups = scale.shape
    if groups != values_per_row // 32:
        raise ValueError(
            f"scale shape {scale.shape} does not cover {values_per_row} values in groups of 32"
        )
    return np.repeat(scale, 2, axis=1)


def to_ggml_nvfp4(packed: np.ndarray, scale_e4m3: np.ndarray) -> np.ndarray:
    """Dense compressed-tensors packed NVFP4 -> ggml block_nvfp4 rows.

    No value or scale byte changes.  Only the nibble and sub-block ordering is
    regrouped: compressed-tensors packs adjacent elements, while ggml pairs
    element j with j+8 in each 16-value sub-block.
    """
    rows, packed_last = packed.shape
    values = packed_last * 2
    if values % QK:
        raise ValueError(f"K={values} is not a multiple of {QK}")
    if scale_e4m3.shape != (rows, values // QK_SUB):
        raise ValueError(f"scale shape {scale_e4m3.shape} != {(rows, values // QK_SUB)}")

    nib = np.empty((rows, values), np.uint8)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4
    sub = nib.reshape(rows, values // QK_SUB, QK_SUB)
    qs = (sub[:, :, : QK_SUB // 2] | (sub[:, :, QK_SUB // 2 :] << 4)).astype(np.uint8)

    blocks = values // QK
    qs = qs.reshape(rows, blocks, N_SUB * (QK_SUB // 2))
    scales = scale_e4m3.reshape(rows, blocks, N_SUB)
    return np.concatenate((scales, qs), axis=2).reshape(rows, blocks * (N_SUB + QK // 2))


def projection_blob(
    checkpoint: Checkpoint, layer: int, expert: int, role: str
) -> tuple[bytes, float]:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}"
    rows, values = ROW_SHAPES[role]

    packed_data, packed_info = checkpoint.tensor(f"{prefix}.weight_sparse_packed")
    mask_data, mask_info = checkpoint.tensor(f"{prefix}.weight_sparse_mask")
    scale_data, scale_info = checkpoint.tensor(f"{prefix}.weight_scale")
    expected = {
        "weight_sparse_packed": (packed_info, "U8", (rows, values // 4)),
        "weight_sparse_mask": (mask_info, "U8", (rows, values // 16)),
        "weight_scale": (scale_info, "F8_E4M3", (rows, values // 32)),
    }
    for name, (info, dtype, shape) in expected.items():
        if info.get("dtype") != dtype or tuple(info.get("shape", ())) != shape:
            raise ValueError(f"{prefix}.{name}: expected {dtype} {shape}, got {info}")

    sparse = np.frombuffer(packed_data, dtype=np.uint8).reshape(rows, values // 4)
    mask = np.frombuffer(mask_data, dtype=np.uint8).reshape(rows, values // 16)
    scale32 = np.frombuffer(scale_data, dtype=np.uint8).reshape(rows, values // 32)
    dense = decompress_pair_bitmask(sparse, mask)
    scale16 = expand_scale_32_to_16(scale32, values)
    ggml = to_ggml_nvfp4(dense, scale16)
    if ggml.shape != (rows, values // QK * (N_SUB + QK // 2)):
        raise AssertionError(f"{prefix}: unexpected GGML shape {ggml.shape}")
    global_divisor = checkpoint.scalar_f32(f"{prefix}.weight_global_scale")
    if not math.isfinite(global_divisor) or global_divisor == 0:
        raise ValueError(
            f"{prefix}.weight_global_scale must be finite and nonzero, got {global_divisor}"
        )
    # MoESQ's effective scale is hard_scale / weight_global_scale; Strata's tail
    # is a multiplier, so store the reciprocal rather than the divisor itself.
    return ggml.tobytes(), 1.0 / global_divisor


def validate_requested_geometry(checkpoint: Checkpoint, layers: int, experts: int) -> None:
    """Validate every requested tensor's dtype and shape before changing output."""
    for layer in range(layers):
        for expert in range(experts):
            for role in ROLES:
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}"
                rows, values = ROW_SHAPES[role]
                wanted = {
                    "weight_sparse_packed": ("U8", (rows, values // 4)),
                    "weight_sparse_mask": ("U8", (rows, values // 16)),
                    "weight_scale": ("F8_E4M3", (rows, values // 32)),
                    "weight_global_scale": ("F32", (1,)),
                }
                for suffix, (dtype, shape) in wanted.items():
                    _, info = checkpoint.tensor(f"{prefix}.{suffix}")
                    if info.get("dtype") != dtype or tuple(info.get("shape", ())) != shape:
                        raise ValueError(f"{prefix}.{suffix}: expected {dtype} {shape}, got {info}")


def expert_blob(checkpoint: Checkpoint, layer: int, expert: int) -> bytes:
    blobs = []
    scales = []
    for role in ROLES:
        blob, scale = projection_blob(checkpoint, layer, expert, role)
        blobs.append(blob)
        scales.append(scale)
    return b"".join(blobs) + struct.pack("<4f", scales[0], scales[1], scales[2], 0.0)


def published_hashes(root: Path) -> dict[str, str]:
    path = root / "SHARD_HASHES.sha256"
    if not path.is_file():
        raise FileNotFoundError(f"missing published shard hashes: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].endswith(".safetensors"):
            result[fields[1]] = fields[0]
    expected = FULL_LAYERS + 1
    if len(result) != expected:
        raise ValueError(f"{path.name}: expected {expected} shard hashes, got {len(result)}")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def base_manifest(base: Path) -> dict[str, Any]:
    missing = [name for name in BASE_FILES if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError("base pack is missing " + ", ".join(missing))
    tokenizer = base / "tokenizer"
    required_tokenizer = (tokenizer / "vocab.json", tokenizer / "chat_template.jinja")
    if any(not path.is_file() for path in required_tokenizer):
        raise FileNotFoundError(
            "base pack is missing tokenizer/vocab.json or tokenizer/chat_template.jinja"
        )
    files = {
        path.relative_to(base).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in [*(base / name for name in BASE_FILES), *sorted(tokenizer.rglob("*"))]
        if path.is_file()
    }
    return {"path": str(base), "files": files}


def native_experts_text(layers: int, experts: int) -> str:
    total = layers * experts * BLOB_BYTES
    head = (
        "# strata native experts v3: layer gu_type d_type offset blob_bytes "
        "gate_off up_off down_off "
        f"(n_expert {experts}, total {total}; offsets in experts.bin)\n"
    )
    lines = []
    for layer in range(layers):
        layer_offset = layer * experts * BLOB_BYTES
        lines.append(
            f"{layer} {NVFP4_TYPE} {NVFP4_TYPE} {layer_offset} {BLOB_BYTES} "
            f"{layer_offset} {layer_offset + GGML_ROW_BYTES} "
            f"{layer_offset + 2 * GGML_ROW_BYTES}\n"
        )
    return head + "".join(lines)


def sidecar(
    checkpoint: Checkpoint,
    layers: int,
    experts: int,
    layout: str,
    source_revision: str | None,
    base: Path | None,
) -> dict[str, Any]:
    hashes = published_hashes(checkpoint.root)
    incomplete_markers = sorted(path.name for path in checkpoint.root.glob("*.aria2"))
    snapshot_complete = not incomplete_markers
    if layers == FULL_LAYERS and experts == FULL_EXPERTS and not snapshot_complete:
        raise ValueError(
            "cannot publish a full pack from an incomplete snapshot: "
            + ", ".join(incomplete_markers)
        )
    return {
        "schema": 2,
        "tool": "tools/moesq_pack.py",
        "source_checkpoint": str(checkpoint.root),
        "source_revision": source_revision,
        "source_hash_authority": "SHARD_HASHES.sha256",
        "snapshot_complete": snapshot_complete,
        "snapshot_validation": "complete published manifest"
        if snapshot_complete
        else "requested headers and tensors only",
        "base_pack": base_manifest(base) if base else None,
        "shards": [
            {
                "name": f"model-{i:05d}-of-{FULL_LAYERS + 1:05d}.safetensors",
                "size": (checkpoint.root / f"model-{i:05d}-of-{FULL_LAYERS + 1:05d}.safetensors")
                .stat()
                .st_size,
                "sha256": hashes[f"model-{i:05d}-of-{FULL_LAYERS + 1:05d}.safetensors"],
            }
            for i in range(1, FULL_LAYERS + 2)
        ],
        "layers": layers,
        "experts_per_layer": experts,
        "partial": layers != FULL_LAYERS or experts != FULL_EXPERTS,
        "native_experts_sha256": hashlib.sha256(layout.encode("utf-8")).hexdigest(),
        "bytes": layers * experts * BLOB_BYTES,
        "weight_contract": {
            "source": "paired-4:8 sparse NVFP4; effective scale is E4M3 / weight_global_scale",
            "destination": "ggml block_nvfp4 plus a per-projection multiplier tail",
            "codes_exact": True,
            "scales_exact": True,
            "global_multiplier": "reciprocal(weight_global_scale), rounded to F32",
            "global_multiplier_exact": False,
            "structured_sparsity_preserved": False,
            "activation_contract": "Strata W4A8 Q8_1; MoESQ input_global_scale is not applied",
        },
    }


def replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def copy_base_files(base: Path, out: Path) -> None:
    base_manifest(base)
    for name in BASE_FILES:
        tmp = out / (name + ".tmp")
        shutil.copyfile(base / name, tmp)
        tmp.replace(out / name)
    tmp_tree = out / "tokenizer.tmp"
    if tmp_tree.exists():
        shutil.rmtree(tmp_tree)
    shutil.copytree(base / "tokenizer", tmp_tree)
    if (out / "tokenizer").exists():
        shutil.rmtree(out / "tokenizer")
    tmp_tree.replace(out / "tokenizer")


def write_pack(
    checkpoint: Checkpoint,
    out: Path,
    base: Path | None,
    layers: int,
    experts: int,
    source_revision: str | None,
) -> None:
    full = layers == FULL_LAYERS and experts == FULL_EXPERTS
    if full and base is None:
        raise ValueError("a full pack needs --base for index.txt, dense.bin, and tokenizer files")
    if not full and base is not None:
        raise ValueError("--base is for a complete 48x512 pack; omit it for a parity prefix")
    if base is not None:
        base_manifest(base)

    out.mkdir(parents=True, exist_ok=True)
    (out / "native_experts.txt").unlink(missing_ok=True)
    part = out / "experts.bin.tmp"
    part.unlink(missing_ok=True)

    layout = native_experts_text(layers, experts)
    written = 0
    with part.open("wb") as stream:
        for layer in range(layers):
            for expert in range(experts):
                blob = expert_blob(checkpoint, layer, expert)
                if len(blob) != BLOB_BYTES:
                    raise AssertionError(f"layer {layer} expert {expert}: wrote {len(blob)} bytes")
                stream.write(blob)
                written += len(blob)
                if written % (256 * BLOB_BYTES) == 0:
                    print(f"  {written / (1 << 30):.2f} GiB", flush=True)
    part.replace(out / "experts.bin")

    replace_text(
        out / "experts.bin.src.json",
        json.dumps(sidecar(checkpoint, layers, experts, layout, source_revision, base), indent=2)
        + "\n",
    )
    if full:
        assert base is not None
        copy_base_files(base, out)
        replace_text(out / "native_experts.txt", layout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="MoESQ safetensors snapshot directory"
    )
    parser.add_argument("--out", required=True, type=Path, help="task-owned output pack directory")
    parser.add_argument(
        "--base", type=Path, help="complete Strata base pack reused by a full output"
    )
    parser.add_argument(
        "--layers", type=int, default=FULL_LAYERS, help="leading layers to write (parity prefix: 1)"
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=FULL_EXPERTS,
        help="leading experts per layer (parity prefix: 1)",
    )
    parser.add_argument(
        "--source-revision", help="published source revision recorded in the sidecar"
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="replace a previous task-owned experts.bin"
    )
    args = parser.parse_args()

    if not 1 <= args.layers <= FULL_LAYERS or not 1 <= args.experts <= FULL_EXPERTS:
        parser.error(f"--layers and --experts must be in 1..{FULL_LAYERS} and 1..{FULL_EXPERTS}")
    full = args.layers == FULL_LAYERS and args.experts == FULL_EXPERTS
    if full and not args.source_revision:
        parser.error("--source-revision is required for a complete pack")
    experts_bin = args.out / "experts.bin"
    if experts_bin.exists() and not args.rewrite:
        raise FileExistsError(
            f"{experts_bin} already exists; use --rewrite only for this task-owned output"
        )

    checkpoint = Checkpoint(args.checkpoint, args.layers, args.experts)
    validate_requested_geometry(checkpoint, args.layers, args.experts)
    write_pack(
        checkpoint,
        args.out,
        args.base.resolve() if args.base else None,
        args.layers,
        args.experts,
        args.source_revision,
    )
    total = args.layers * args.experts * BLOB_BYTES
    print(
        f"MoESQ experts: {args.layers} layer(s) x {args.experts} expert(s), "
        f"{total} bytes ({total / 2**30:.3f} GiB), type {NVFP4_TYPE}/{NVFP4_TYPE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
