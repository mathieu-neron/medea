"""Small PyTorch MLP head for the fused 1296-dim feature vector.

Architecture: 1296 → 128 → 64 → 1 logit, with LayerNorm + GELU + Dropout
between hidden layers. Tiny by design — with 94 samples and 1296 features,
the model will overfit easily; capacity has to be small enough that L2 weight
decay + dropout + early stopping can keep it honest.

LayerNorm (not BatchNorm) because the LOGO folds give batch sizes ≤ 32 and
BatchNorm's running stats wobble badly at that scale. LayerNorm normalizes
along the feature dim per-sample and behaves identically train/eval.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

INPUT_DIM = 1296
DEFAULT_HIDDEN = (128, 64)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits; apply sigmoid externally for probabilities.
        return self.net(x).squeeze(-1)
