from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from .constants import MLP_USE_LAYER_NORM
except ImportError:
    from constants import MLP_USE_LAYER_NORM


class MLPErrorModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_p: float = 0.08, use_layer_norm: bool = MLP_USE_LAYER_NORM) -> None:
        super().__init__()
        safe_dropout = float(np.clip(dropout_p, 0.0, 0.5))
        self.use_layer_norm = bool(use_layer_norm)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, 128),
            self._build_norm(128),
            nn.Tanh(),
            nn.Dropout(safe_dropout),
            nn.Linear(128, output_dim),
        )
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _build_norm(self, hidden_dim: int) -> nn.Module:
        if self.use_layer_norm:
            return nn.LayerNorm(hidden_dim)
        return nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
