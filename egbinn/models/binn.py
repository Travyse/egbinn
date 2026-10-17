"""BINN: non-ensemble sparse biological hierarchy aging clock.

A stack of sparse masked linear layers (plain or gated) maps CpG beta values
through a biological hierarchy (cpg->gene->L2->L1->L0). An activation follows
every stage, and each stage's hidden state feeds its own linear age head.
The per-head predictions are stacked along dim 1, ``(B, H, 1)``; the training
adapter averages them into a single prediction.
"""

__all__ = ["BINNRegression"]

import torch
from torch import nn

from .sparse import GatedSparseLinearLayer, SparseMaskedLinear

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}


class BINNRegression(nn.Module):
    """Plain (non-ensemble) BINN aging clock with multi-stage sparse hierarchy.

    Parameters
    ----------
    input_dim : int
        Number of input features (CpG beta values).
    stages : list[dict]
        Ordered list of stage configs, each containing:
          - mask_indices: tensor (2, n_edges) int64
          - out_features: int
          - layer_type: "sparse" | "gated"
          - gate_activation: str (for gated layers, "silu" or "sigmoid")
          - match_dense_init: bool
          - pos_weights: bool
          - value_bias: bool (gated) / bias: bool (sparse)
          - gate_bias: bool (gated only)
          - gate_bias_init: float or None (gated only)
          - gate_weight_init: str (gated only, "default"|"zero"|"scaled")
          - gate_weight_scale: float (gated only, used when gate_weight_init="scaled")
    stage_activation : str, default "relu"
        Activation applied after every sparse stage, before its head.
        One of "relu", "silu", "gelu", "tanh", "none" (identity).
    dropout : float
        Dropout applied after each stage activation.
    input_dropout : float or None
        Dropout on raw input features before the first stage.
    gate_weight_init : str, default "default"
        Broadcast to all gated stages as the default ``gate_weight_init``.
    gate_weight_scale : float, default 0.01
        Broadcast to all gated stages as the default ``gate_weight_scale``.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        stages: list[dict] | None = None,
        stage_activation: str = "relu",
        dropout: float = 0.0,
        input_dropout: float | None = None,
        gate_weight_init: str = "default",
        gate_weight_scale: float = 0.01,
    ):
        super().__init__()
        if input_dim == 0:
            raise ValueError("Input dimension must be greater than 0.")
        if stages is None or len(stages) == 0:
            raise ValueError("stages must be a non-empty list of stage configs.")
        if stage_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown stage_activation: {stage_activation!r}. Expected one of {sorted(_ACTIVATIONS)}.")

        self.stage_activation = stage_activation

        self.input_dropout_layer = nn.Dropout(input_dropout) if input_dropout is not None else None

        self.sparse_stages = nn.ModuleList()
        self.heads = nn.ModuleList()
        running_in = input_dim
        for i, stage in enumerate(stages):
            mask_indices = stage["mask_indices"]
            layer_type = stage.get("layer_type", "sparse")
            match_dense_init = stage.get("match_dense_init", True)
            pos_weights = stage.get("pos_weights", False)
            out_features = stage["out_features"]

            if layer_type == "sparse":
                bias = stage.get("bias", stage.get("value_bias", True))
                layer = SparseMaskedLinear(
                    mask_indices=mask_indices,
                    in_features=running_in,
                    out_features=out_features,
                    bias=bias,
                    match_dense_init=match_dense_init,
                    pos_weights=pos_weights,
                )
            elif layer_type == "gated":
                gate_activation = stage.get("gate_activation", "silu")
                value_bias = stage.get("value_bias", True)
                gate_bias = stage.get("gate_bias", False)
                gate_bias_init = stage.get("gate_bias_init", None)
                stage_gate_weight_init = stage.get("gate_weight_init", gate_weight_init)
                stage_gate_weight_scale = stage.get("gate_weight_scale", gate_weight_scale)
                layer = GatedSparseLinearLayer(
                    mask_indices=mask_indices,
                    in_features=running_in,
                    out_features=out_features,
                    value_bias=value_bias,
                    gate_bias=gate_bias,
                    match_dense_init=match_dense_init,
                    pos_weights=pos_weights,
                    gate_activation=gate_activation,
                    gate_bias_init=gate_bias_init,
                    gate_weight_init=stage_gate_weight_init,
                    gate_weight_scale=stage_gate_weight_scale,
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type!r} at stage {i}")

            self.sparse_stages.append(layer)
            self.heads.append(nn.Linear(out_features, 1))
            running_in = out_features

        self.activation = _ACTIVATIONS[stage_activation]()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, batch) -> torch.Tensor:
        x = batch["betas"]

        if self.input_dropout_layer is not None:
            x = self.input_dropout_layer(x)

        preds = []
        for stage, head in zip(self.sparse_stages, self.heads, strict=True):
            x = stage(x)
            x = self.activation(x)
            x = self.dropout_layer(x)
            preds.append(head(x))
        return torch.stack(preds, dim=1)  # (B, H, 1)
