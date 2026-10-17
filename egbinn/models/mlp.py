__all__ = ["MLPRegression"]

import torch.nn as nn

from .common import MLP


class MLPRegression(nn.Module):
    """Vanilla MLP regression baseline operating on a flat beta vector."""

    def __init__(
        self,
        input_dim=1024,
        hidden_size=32,
        num_layers=2,
        input_dropout=None,
        dropout=0.0,
    ):
        super().__init__()
        if input_dim == 0:
            raise ValueError("Input dimension must be greater than 0.")
        self.hidden_size = hidden_size
        self.mlp = MLP(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_dropout=input_dropout,
            dropout=dropout,
        )
        self.task_head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        x = self.mlp(batch["betas"])
        y = self.task_head(x)
        return x, y
