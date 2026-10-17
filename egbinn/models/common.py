__all__ = [
    "MLP",
    "BaseMLPModel",
    "LinearEfficientEnsemble",
    "LowRankLinear",
    "NLinear",
    "ScaleEnsemble",
    "TabMMLP",
    "init_random_signs_",
    "init_rsqrt_uniform_",
    "make_efficient_ensemble",
]

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import Parameter


class LowRankLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=bias)

    def forward(self, x):
        return self.up(self.down(x))


class BaseMLPModel(nn.Module):
    def __init__(self, input_dropout: float | None = None, dropout: float | None = None):
        super().__init__()
        self.dropout = float(dropout) if dropout is not None else 0.0
        self.input_dropout = self.dropout if input_dropout is None else float(input_dropout)

    def get_activation_fn(self, activation_fn):
        if activation_fn == "relu":
            return nn.ReLU()
        elif activation_fn == "silu":
            return nn.SiLU()
        elif activation_fn == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")


class MLP(BaseMLPModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        input_dropout: float | None = None,
        dropout: float = 0.0,
        layer_norm: bool = True,
        use_low_rank: bool = False,
        activation_fn: str = "relu",
    ):
        super().__init__(input_dropout=input_dropout, dropout=dropout)
        self.layers = nn.ModuleList()

        self.input_dropout_layer = nn.Dropout(self.input_dropout)
        self.layers.append(
            nn.Sequential(
                LowRankLinear(input_size, hidden_size, rank=hidden_size // 2)
                if use_low_rank
                else nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
                self.get_activation_fn(activation_fn),
                nn.Dropout(self.dropout),
            )
        )

        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
                    self.get_activation_fn(activation_fn),
                    nn.Dropout(self.dropout),
                )
            )

    def forward(self, x):
        x = self.input_dropout_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


class NLinear(nn.Module):
    """A stack of N independent linear layers.

    Each layer is applied to its own slice of a 3D input.

    Shape:
        - Input:  ``(B, N, in_features)``
        - Output: ``(B, N, out_features)``
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class ScaleEnsemble(nn.Module):
    """Per-member multiplicative scaling for the ensemble input adapter (tabm-mini)."""

    def __init__(self, k: int, d: int, *, init: Literal["ones", "normal", "random-signs"]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self):
        if self._weight_init == "ones":
            nn.init.ones_(self.weight)
        elif self._weight_init == "normal":
            nn.init.normal_(self.weight)
        elif self._weight_init == "random-signs":
            init_random_signs_(self.weight)
        else:
            raise ValueError(f"Unknown weight_init: {self._weight_init}")

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """BatchEnsemble layer.

    One shared weight matrix plus per-member multiplicative scalings
    (``r`` on input, ``s`` on output) and optional per-member bias.

    Shape:
        - Input:  ``(B, K, in_features)``
        - Output: ``(B, K, out_features)``
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal["ones", "random-signs"],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            "r",
            (nn.Parameter(torch.empty(k, in_features)) if ensemble_scaling_in else None),
        )
        self.register_parameter(
            "s",
            (nn.Parameter(torch.empty(k, out_features)) if ensemble_scaling_out else None),
        )
        self.register_parameter(
            "bias",
            (
                nn.Parameter(torch.empty(out_features))
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init
        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {"ones": nn.init.ones_, "random-signs": init_random_signs_}[self.scaling_init]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        if self.bias is not None:
            x = x + self.bias
        return x


def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    """Recursively replace ``nn.Linear`` children with ``EnsembleLayer`` in-place."""
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


class TabMMLP(nn.Module):
    """Simple MLP backbone for TabM (no LayerNorm).

    Blocks of ``Linear -> activation -> Dropout``. Operates on both 2D ``(B, D)``
    and 3D ``(B, K, D)`` inputs.
    """

    def __init__(
        self,
        d_in: None | int = None,
        d_block: int = 32,
        n_blocks: int = 5,
        dropout: float = 0.0,
        activation: str = "ReLU",
        first_block: nn.Module | None = None,
    ):
        super().__init__()
        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList()
        running_in = d_first
        for i in range(n_blocks):
            if i == 0 and first_block is not None:
                layer = first_block
                running_in = first_block.out_features
            else:
                layer = nn.Linear(running_in, d_block)  # ty: ignore[invalid-argument-type]
                running_in = d_block
            self.blocks.append(
                nn.Sequential(
                    layer,
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
