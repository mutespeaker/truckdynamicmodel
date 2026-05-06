from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from .constants import MLP_DROPOUT_P, MLP_HIDDEN_DIM, MLP_HIDDEN_LAYERS, MLP_USE_LAYER_NORM
except ImportError:
    from constants import MLP_DROPOUT_P, MLP_HIDDEN_DIM, MLP_HIDDEN_LAYERS, MLP_USE_LAYER_NORM


class MLPErrorModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_p: float = MLP_DROPOUT_P,
        use_layer_norm: bool = MLP_USE_LAYER_NORM,
        hidden_dim: int = MLP_HIDDEN_DIM,
        hidden_layers: int = MLP_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        safe_dropout = float(np.clip(dropout_p, 0.0, 0.5))
        self.use_layer_norm = bool(use_layer_norm)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {self.hidden_layers}")

        layers: list[nn.Module] = []
        layer_input_dim = int(input_dim)
        for _ in range(self.hidden_layers):
            layers.extend(
                [
                    nn.Linear(layer_input_dim, self.hidden_dim),
                    self._build_norm(self.hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(safe_dropout),
                ]
            )
            layer_input_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _build_norm(self, hidden_dim: int) -> nn.Module:
        if self.use_layer_norm:
            return nn.LayerNorm(hidden_dim)
        return nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
