"""EGBINN: Ensembled Gated Biological Interpretable Neural Network.

A stack of gated sparse masked linear layers maps CpG beta values through a
biological hierarchy (cpg->gene->L2->L1->L0), followed by optional
ensemblified dense blocks and a TabM ensemble head.

The first stage is shared across all ``k`` ensemble members (biological edge
structure is not ensemblified). Subsequent stages are ensemblified via
BatchEnsemble adapters (``r`` / ``s`` / ``ensemble_bias``) inside each sparse
layer.
"""

__all__ = ["EGBINN"]

import torch
import torch.nn as nn

from .common import LinearEfficientEnsemble, NLinear, init_random_signs_
from .sparse import GatedSparseLinearLayer, SparseMaskedLinear

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}


class EGBINN(nn.Module):
    """EGBINN aging clock with multi-stage gated sparse hierarchy + TabM ensemble.

    Parameters
    ----------
    input_dim : int
        Number of input features (CpG beta values).
    stages : list[dict]
        Ordered list of stage configs, each containing:
          - mask_indices: tensor (2, n_edges) int64
          - out_features: int
          - layer_type: "sparse" | "gated"
          - ensemblify: bool (whether to apply BatchEnsemble)
          - gate_activation: str (for gated layers, "silu" or "sigmoid")
          - match_dense_init: bool
          - pos_weights: bool
          - value_bias: bool (gated) / bias: bool (sparse)
          - gate_bias: bool (gated only)
          - gate_bias_init: float or None (gated only)
          - gate_weight_init: str (gated only, "default"|"zero"|"scaled")
          - gate_weight_scale: float (gated only, used when gate_weight_init="scaled")
    k : int
        Ensemble member count.
    d_block_mult : float, default 1.0
        Multiplier for the hidden dim of the dense blocks after the sparse
        stack. The actual ``d_block`` is derived as
        ``max(1, int(d_block_mult * l0_dim))`` where ``l0_dim`` is the
        ``out_features`` of the last sparse stage. Ignored when
        ``n_dense_blocks=0``.
    n_dense_blocks : int, default 1
        Number of ensemblified dense blocks after the sparse stack.
        0 = direct L0->head (head is NLinear(k, l0, 1)).
    dropout : float
        Dropout applied after each sparse stage activation and inside
        dense blocks.
    input_dropout : float or None
        Dropout on raw input features before the ensemble expand.
    share_training_batches : bool, default True
        If True, replicate batch across ``k`` during training. If False,
        the loader is expected to supply ``batch_size * k`` samples.
    stage_activation : str, default "none"
        Activation applied after every sparse stage. Default "none" relies
        on the gate SiLU for nonlinearity. Set to "relu" (or similar) when
        using all-sparse stages.
    regression_approach : str, default "final"
        How age predictions are produced. ``"final"`` uses a single
        ``NLinear(k, head_in, 1)`` head on the final representation and
        returns ``(B, K, 1)``. ``"multi_stage"`` attaches a per-stage head
        to every sparse stage's hidden state and the final dense-block
        output (if ``n_dense_blocks > 0``), returning ``(B, H, K, 1)``.
    gate_weight_init : str, default "default"
        Broadcast to all gated stages as the default ``gate_weight_init``.
    gate_weight_scale : float, default 0.01
        Broadcast to all gated stages as the default ``gate_weight_scale``.
    """

    _REGRESSION_APPROACHES = ("final", "multi_stage")

    def __init__(
        self,
        input_dim: int = 1024,
        stages: list[dict] | None = None,
        k: int = 16,
        d_block_mult: float = 1.0,
        n_dense_blocks: int = 1,
        dropout: float = 0.0,
        input_dropout: float | None = None,
        share_training_batches: bool = True,
        stage_activation: str = "none",
        regression_approach: str = "final",
        gate_weight_init: str = "default",
        gate_weight_scale: float = 0.01,
    ):
        super().__init__()
        if input_dim == 0:
            raise ValueError("Input dimension must be greater than 0.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if stages is None or len(stages) == 0:
            raise ValueError("stages must be a non-empty list of stage configs.")
        if stage_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown stage_activation: {stage_activation!r}. Expected one of {sorted(_ACTIVATIONS)}.")
        if regression_approach not in self._REGRESSION_APPROACHES:
            raise ValueError(
                f"Unknown regression_approach: {regression_approach!r}. "
                f"Expected one of {sorted(self._REGRESSION_APPROACHES)}."
            )

        self.k = k
        self.share_training_batches = share_training_batches
        self.n_dense_blocks = n_dense_blocks
        self.d_block_mult = d_block_mult
        self.stage_activation = stage_activation
        self.regression_approach = regression_approach

        self.input_dropout_layer = nn.Dropout(input_dropout) if input_dropout is not None else None

        self.sparse_stages = nn.ModuleList()
        stage_out_dims: list[int] = []
        running_in = input_dim
        for i, stage in enumerate(stages):
            mask_indices = stage["mask_indices"]
            layer_type = stage.get("layer_type", "gated")
            ensemblify = stage.get("ensemblify", i > 0)
            stage_k = k if ensemblify else None
            match_dense_init = stage.get("match_dense_init", True)
            pos_weights = stage.get("pos_weights", False)
            out_features = stage["out_features"]
            stage_out_dims.append(out_features)

            if layer_type == "sparse":
                bias = stage.get("bias", stage.get("value_bias", True))
                layer = SparseMaskedLinear(
                    mask_indices=mask_indices,
                    in_features=running_in,
                    out_features=out_features,
                    bias=bias,
                    match_dense_init=match_dense_init,
                    pos_weights=pos_weights,
                    k=stage_k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init="ones",
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
                    k=stage_k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init="ones",
                    gate_bias_init=gate_bias_init,
                    gate_weight_init=stage_gate_weight_init,
                    gate_weight_scale=stage_gate_weight_scale,
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type!r} at stage {i}")

            self.sparse_stages.append(layer)
            running_in = out_features

        l0_dim = running_in
        d_block = max(1, int(d_block_mult * l0_dim))
        self.d_block = d_block

        self.dense_blocks = nn.ModuleList()
        block_in = l0_dim
        for _ in range(n_dense_blocks):
            self.dense_blocks.append(
                nn.Sequential(
                    LinearEfficientEnsemble(
                        in_features=block_in,
                        out_features=d_block,
                        bias=True,
                        k=k,
                        ensemble_scaling_in=True,
                        ensemble_scaling_out=True,
                        ensemble_bias=True,
                        scaling_init="ones",
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            block_in = d_block

        head_in = d_block if n_dense_blocks > 0 else l0_dim
        self.output = NLinear(k, head_in, 1)

        if regression_approach == "multi_stage":
            self.stage_heads = nn.ModuleList([NLinear(k, dim, 1) for dim in stage_out_dims])
            if n_dense_blocks > 0:
                self.stage_heads.append(NLinear(k, d_block, 1))
            self.n_heads = len(self.stage_heads)
        else:
            self.stage_heads = nn.ModuleList()
            self.n_heads = 0

        self._init_first_ensemble_random_signs()

        self.activation = _ACTIVATIONS[stage_activation]()
        self.stage_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _init_first_ensemble_random_signs(self) -> None:
        for stage in self.sparse_stages:
            r = getattr(stage, "r", None)
            if stage.k is not None and r is not None and isinstance(r, torch.Tensor):
                init_random_signs_(r)
                return
        for block in self.dense_blocks:
            children = list(block.children())
            first = children[0]
            r = getattr(first, "r", None)
            if isinstance(first, LinearEfficientEnsemble) and r is not None and isinstance(r, torch.Tensor):
                init_random_signs_(r)
                return

    def forward(self, batch) -> torch.Tensor:
        x = batch["betas"]

        if self.input_dropout_layer is not None:
            x = self.input_dropout_layer(x)

        if self.share_training_batches or not self.training:
            x = x[:, None].expand(-1, self.k, -1)
        else:
            x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])

        if self.regression_approach == "multi_stage":
            stage_hidden = []
            for stage in self.sparse_stages:
                x = stage(x)
                x = self.activation(x)
                x = self.stage_dropout(x)
                stage_hidden.append(x)
            for block in self.dense_blocks:
                x = block(x)
            if self.n_dense_blocks > 0:
                stage_hidden.append(x)
            preds = [head(h) for head, h in zip(self.stage_heads, stage_hidden, strict=True)]
            return torch.stack(preds, dim=1)  # (B, H, K, 1)

        for stage in self.sparse_stages:
            x = stage(x)
            x = self.activation(x)
            x = self.stage_dropout(x)
        for block in self.dense_blocks:
            x = block(x)
        x = self.output(x)
        return x
