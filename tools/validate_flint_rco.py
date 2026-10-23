#!/usr/bin/env python3
"""Run a tiny Flint model through stock RCO's Gumbel-STE optimizer."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import flint_rco_adapter as adapter  # noqa: E402

DEFAULT_RCO = Path.home() / ".local/share/strata-tools/RCO"


def load_rco_prune(root: Path):
    path = root / "src/search/prune.py"
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    loader = importlib.machinery.SourceFileLoader("strata_rco_flint_probe", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyExperts(nn.Module):
    def __init__(self, experts: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, hidden) * 0.05)

    def forward(self, hidden, indices, weights):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_indices = indices.reshape(-1)
        selected = self.weight.index_select(0, flat_indices)
        weighted = selected * weights.reshape(-1, 1).to(selected.dtype)
        output = weighted.sum(dim=0, keepdim=True).expand_as(flat_hidden)
        return output.reshape(hidden.shape)


class TinyMoe(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden, experts, bias=True)
        self.experts = TinyExperts(experts, hidden)
        self.shared_expert = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden):
        scores = self.gate(hidden).float().sigmoid()
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = self.experts(hidden, indices, weights.to(hidden.dtype))
        return routed + self.shared_expert(hidden)


class TinyDecoder(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden):
        return self.mlp(hidden)


class TinyFlintLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=5,
            mlp_only_layers=[0, 1],
            num_experts=4,
            num_experts_per_tok=2,
        )
        self.model = SimpleNamespace(
            layers=nn.ModuleList(
                [
                    TinyDecoder(nn.Linear(16, 16, bias=False)),
                    TinyDecoder(nn.Linear(16, 16, bias=False)),
                    TinyDecoder(TinyMoe(16, 4, 2)),
                    TinyDecoder(TinyMoe(16, 4, 2)),
                    TinyDecoder(TinyMoe(16, 4, 2)),
                ]
            )
        )
        self.embed_tokens = nn.Embedding(32, 16)
        self.lm_head = nn.Linear(16, 32, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, use_cache=False):
        del use_cache
        hidden = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = hidden + layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def run(rco_root: Path, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    prune = load_rco_prune(rco_root)
    model = TinyFlintLM().eval()
    counts = adapter.get_moe_info(model)
    if counts != (3, 4, 2):
        raise AssertionError(f"unexpected counts: {counts}")

    calibration = torch.randint(0, 32, (2, 8))
    masks = torch.ones_like(calibration)
    references, reference_masks = prune.build_ref_cache(model, calibration, masks, 1)
    alpha = torch.zeros(counts.n_layers * counts.n_experts, 2, requires_grad=True)
    ste_keep, assignment = prune.compute_ste_masks(alpha, tau=1.0, budget=3)
    shared_state = {"ste_masks": ste_keep.view(counts.n_layers, counts.n_experts)}
    wrappers = adapter.install_wrappers(model, counts.n_layers, counts.n_experts, shared_state)
    try:
        loss = prune.compute_kl_loss(
            model,
            calibration[:1],
            references[0],
            mask=reference_masks[0],
            topk=8,
        )
        loss.backward()
    finally:
        adapter.remove_wrappers(wrappers)
    if alpha.grad is None or not bool((alpha.grad.abs() > 1e-12).any()):
        raise AssertionError("stock RCO KL loss produced no alpha gradient")
    search_mask = assignment.view(counts.n_layers, counts.n_experts).bool()
    kl = adapter.evaluate_with_mask(model, search_mask, calibration, references, reference_masks, 1)

    return {
        "status": "PASS",
        "rco_root": str(rco_root),
        "seed": seed,
        "counts": list(counts),
        "target_pruned_experts": 3,
        "actual_pruned_experts": int(search_mask.sum().item()),
        "hard_mask_kl": kl,
        "alpha_finite": bool(torch.isfinite(alpha).all().item()),
        "stock_kl_loss": loss.item(),
        "alpha_gradient_nonzero": bool(
            alpha.grad is not None and (alpha.grad.abs() > 1e-12).any().item()
        ),
        "physical_keep_rows": adapter.physical_keep_rows(search_mask),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rco-root", type=Path, default=DEFAULT_RCO)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args.rco_root, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
