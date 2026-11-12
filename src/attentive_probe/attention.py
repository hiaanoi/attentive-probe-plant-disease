from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class MILOutput:
    logits: Tensor
    attention: Tensor
    residual_logits: Tensor

class ClassConditionedMIL(nn.Module):
    """Small residual attention-MIL head on frozen visual features."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 115,
        mode: Literal["shared", "classwise"] = "classwise",
    ) -> None:
        super().__init__()
        if mode not in {"shared", "classwise"}:
            raise ValueError(f"Unknown mode: {mode}")
        if min(in_dim, hidden_dim, num_classes) <= 0:
            raise ValueError("Dimensions must be positive")
        self.mode = mode
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.key_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.value_net = nn.Linear(in_dim, hidden_dim)
        queries = 1 if mode == "shared" else num_classes
        self.queries = nn.Parameter(torch.empty(queries, hidden_dim))
        self.class_weights = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.queries, std=1.0)
        nn.init.normal_(self.class_weights, std=0.01)

    def forward(self, patches: Tensor, global_logits: Tensor) -> MILOutput:
        if patches.ndim != 3 or global_logits.ndim != 2:
            raise ValueError("patches must be [B,N,D], global_logits [B,C]")
        b, n, _ = patches.shape
        if global_logits.shape != (b, self.num_classes):
            raise ValueError("global_logits has an unexpected shape")
        if n <= 0:
            raise ValueError("No image patches supplied")
        keys = self.key_net(patches)
        values = self.value_net(patches)
        # float32 softmax for AMP stability; query is a learned model parameter.
        scores = torch.einsum("bnh,ch->bcn", keys, self.queries)
        scores = scores.float() / math.sqrt(self.hidden_dim)
        alpha = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
        local = torch.einsum("bqn,bnh->bqh", alpha, values)
        if self.mode == "shared":
            local = local.expand(-1, self.num_classes, -1)
            alpha = alpha.expand(-1, self.num_classes, -1)
        residual = (local * self.class_weights.unsqueeze(0)).sum(-1) + self.class_bias
        return MILOutput(logits=global_logits + residual, attention=alpha, residual_logits=residual)

def lesion_region_mass_loss(
    attention: Tensor,
    labels: Tensor,
    patch_coverage: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Encourage attention mass inside true-class GT lesion, not uniform mask fit.

    labels: [B] class index for each training example.
    patch_coverage: [B,N], fractional lesion occupancy in [0, 1].
    Empty-mask samples are skipped. Only call this during training.
    The region loss is used during training only.
    """
    if attention.ndim != 3 or patch_coverage.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected attention [B,C,N], masks [B,N], labels [B]")
    b, c, n = attention.shape
    if patch_coverage.shape != (b, n) or labels.shape != (b,):
        raise ValueError("Incompatible tensor shapes")
    if (labels < 0).any() or (labels >= c).any():
        raise ValueError("Class label out of range")
    mask = patch_coverage.float()
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('Coverage must be finite and within [0,1]')
    valid = mask.sum(dim=-1) > 0
    if not valid.any():
        # Differentiable zero loss for an all-empty-mask minibatch.
        return attention.sum() * 0.0
    batch_ix = torch.arange(b, device=attention.device)
    true_class_attn = attention.float()[batch_ix, labels.long(), :]
    mass = (true_class_attn * mask).sum(dim=-1)
    return -torch.log((mass[valid] + eps)).mean()

def combined_loss(output: MILOutput, labels: Tensor, patch_coverage: Tensor | None,
                  region_weight: float = 0.0) -> tuple[Tensor, dict[str, float]]:
    ce = F.cross_entropy(output.logits.float(), labels.long())
    if region_weight > 0 and patch_coverage is None:
        raise ValueError("GT lesion coverage is mandatory for the region loss during training")
    aux = (lesion_region_mass_loss(output.attention, labels, patch_coverage)
           if region_weight > 0 and patch_coverage is not None else ce.detach() * 0.0)
    total = ce + region_weight * aux
    return total, {"classification_loss": float(ce.detach()),
                   "region_loss": float(aux.detach()), "total_loss": float(total.detach())}
