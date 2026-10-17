"""Flat-parameter sparse masked linear layer.

A drop-in replacement for XAI-AGE's ``CustomizedLinear``
(``examples/XAI-AGE-pytorch-main/src/pnet/CustomizedLinear.py``) that uses a
flat 1-D weight plus ``index_add_`` so it scales to CpG->gene layers
(~18 GB dense mask -> ~2 MB flat at fold-1 scale: 241,609 CpGs x 19,926 genes,
186,792 biological edges).

Mathematically equivalent to a masked dense ``nn.Linear``:
``out = x @ (W . M).T + bias`` restricted to live edges, where the edge
structure (``cpg_idx``, ``gene_idx``) *is* the mask -- no dense mask tensor is
ever materialized in the forward/backward/optimizer hot path.

Artifacts consumed (per fold, exported by the ``Epigenetic-Dataset-Management``
repo's ``09_map_cpgs_to_genes.ipynb``)::

    mask_indices.npy   (2, n_edges) int64  -- row0=cpg_input_idx, row1=gene_output_idx
    gene_index.csv     gene, gene_idx      -- output-row assignment (alphabetical)
    probe_order.csv    IlmnID              -- input-column order (memmap order)
    shapes.json        in_features, out_features, n_edges, ...

Index convention: edge ``e`` corresponds to ``W[gene_idx[e], cpg_idx[e]]`` of an
``nn.Linear(in_features, out_features)`` weight of shape ``(out_features, in_features)``.
"""

__all__ = ["SparseMaskedLinear", "GatedSparseLinearLayer"]

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_rsqrt_uniform_(x: torch.Tensor, d: int) -> torch.Tensor:
    """Uniform init with bound ``1/sqrt(d)`` (matches LinearEfficientEnsemble)."""
    assert d > 0
    return nn.init.uniform_(x, -(d**-0.5), d**-0.5)


@torch.inference_mode()
def _init_random_signs_(x: torch.Tensor) -> torch.Tensor:
    """Random +-1 init (matches LinearEfficientEnsemble's random-signs option)."""
    return x.bernoulli_(0.5).mul_(2).add_(-1)


class SparseMaskedLinear(nn.Module):
    """Masked linear layer with a flat 1-D trainable weight per biological edge.

    Stores one scalar per live edge (``_weight_flat``, shape ``(n_edges,)``) plus
    optional per-output bias. Forward gathers input features by ``cpg_idx``,
    scales by the edge weight, and scatters into the output via ``index_add_``
    along ``gene_idx``. Backward is handled by autograd -- only live edges
    receive gradient (sparse grad on ``_weight_flat``), matching
    ``CustomizedLinear``'s ``grad_weight *= mask`` without requiring a mask
    tensor.

    Parameters
    ----------
    mask_indices : array_like, shape (2, n_edges), int64
        Edge index pairs. Row 0 = CpG input index (column of ``x``), row 1 =
        gene output index (row of the equivalent dense weight). Accepted as a
        torch tensor, numpy array, or list-of-lists; coerced to ``torch.long``.
    in_features : int
        Number of input features (CpG beta values). ``x.size(-1)`` must equal this.
    out_features : int
        Number of output features (genes). ``gene_idx`` values must be in
        ``[0, out_features)``.
    bias : bool, default True
        If True, adds a learnable per-output bias.
    match_dense_init : bool, default True
        If True, initialize ``_weight_flat`` and ``bias`` from
        ``U(-1/sqrt(in_features), 1/sqrt(in_features))`` -- distributionally
        identical to ``CustomizedLinear.__init__``'s
        ``kaiming_uniform_(a=sqrt(5))`` (verified: that bound equals
        ``1/sqrt(fan_in)``). If False, use a sparse-aware per-gene-degree init:
        each edge ``e`` feeding gene ``g`` is drawn from
        ``U(-1/sqrt(degree(g)), 1/sqrt(degree(g)))``. More physically correct
        for varying gene degrees but not distribution-matched to the dense
        reference.
    pos_weights : bool, default False
        If True, override the chosen mode and sample weights and bias from
        ``U(0, 2*bound)`` (matches ``CustomizedLinear.reset_params_pos``).
    k : int or None, default None
        If set, enables BatchEnsemble mode: per-member input scaling ``r``
        (``k, in_features``), output scaling ``s`` (``k, out_features``), and
        per-member ``ensemble_bias`` (``k, out_features``) are created. When
        None, the layer is a pure sparse primitive (no ensemble params).
    ensemble_scaling_in : bool, default True
        If True (and ``k`` is set), create ``r``.
    ensemble_scaling_out : bool, default True
        If True (and ``k`` is set), create ``s``.
    ensemble_bias : bool, default True
        If True (and ``k`` is set), create per-member bias.
    scaling_init : str, default "ones"
        Init for ``r`` and ``s``: ``"ones"`` or ``"random-signs"``.

    Attributes
    ----------
    cpg_idx : torch.Tensor            (buffer, int64, shape (n_edges,))
    gene_idx : torch.Tensor           (buffer, int64, shape (n_edges,))
    _weight_flat : nn.Parameter       (shape (n_edges,)) -- the only trainable weight
    bias : nn.Parameter or None       (shape (out_features,))
    r : nn.Parameter or None          (shape (k, in_features))  -- ensemble only
    s : nn.Parameter or None          (shape (k, out_features)) -- ensemble only
    ensemble_bias : nn.Parameter or None (shape (k, out_features)) -- ensemble only

    Notes
    -----
    The trainable parameter is named ``_weight_flat`` (not ``weight``) to avoid
    clashing with the ``.weight`` introspection shim (see :attr:`weight`), which
    materializes a dense ``(out_features, in_features)`` tensor (~18 GB at CpG
    scale). Use :meth:`weight_sparse` for a memory-efficient COO view.
    """

    __constants__ = ["in_features", "out_features", "n_edges"]

    cpg_idx: torch.Tensor
    gene_idx: torch.Tensor
    _weight_flat: nn.Parameter
    bias: nn.Parameter | None
    r: nn.Parameter | None
    s: nn.Parameter | None
    ensemble_bias: nn.Parameter | None

    def __init__(
        self,
        mask_indices,
        in_features: int,
        out_features: int,
        bias: bool = True,
        match_dense_init: bool = True,
        pos_weights: bool = False,
        k: int | None = None,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = True,
        scaling_init: Literal["ones", "random-signs"] = "ones",
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        idx = torch.as_tensor(mask_indices, dtype=torch.long)
        if idx.ndim != 2 or idx.size(0) != 2:
            raise ValueError(f"mask_indices must have shape (2, n_edges), got {tuple(idx.shape)}")
        n_edges = idx.size(1)

        cpg_idx = idx[0]
        gene_idx = idx[1]
        if cpg_idx.size(0) != gene_idx.size(0):
            raise ValueError("mask_indices rows must have equal length")
        if n_edges > 0:
            if int(cpg_idx.max().item()) >= in_features:
                raise ValueError("cpg_idx contains an index >= in_features")
            if int(gene_idx.max().item()) >= out_features:
                raise ValueError("gene_idx contains an index >= out_features")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_edges = int(n_edges)
        self.match_dense_init = bool(match_dense_init)
        self.pos_weights = bool(pos_weights)
        self.k = k

        self.register_buffer("cpg_idx", cpg_idx)
        self.register_buffer("gene_idx", gene_idx)

        self._weight_flat = nn.Parameter(torch.empty(n_edges))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.r: nn.Parameter | None = None
        self.s: nn.Parameter | None = None
        self.ensemble_bias: nn.Parameter | None = None
        if k is not None:
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            if ensemble_scaling_in:
                self.r = nn.Parameter(torch.empty(k, in_features))
            if ensemble_scaling_out:
                self.s = nn.Parameter(torch.empty(k, out_features))
            if ensemble_bias:
                self.ensemble_bias = nn.Parameter(torch.empty(k, out_features))
        self._scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-sample ``_weight_flat`` (and ``bias``) from a uniform distribution.

        With ``match_dense_init=True`` the bound is ``1/sqrt(in_features)``,
        matching ``CustomizedLinear.__init__``'s
        ``kaiming_uniform_(a=sqrt(5))`` (which yields the same bound). With
        ``match_dense_init=False`` each edge uses ``1/sqrt(degree(gene))``.
        ``pos_weights=True`` shifts the range to ``U(0, 2*bound)``.
        """
        if self.pos_weights:
            self._reset_pos()
        else:
            self._reset_standard()
        self._reset_ensemble()

    def _reset_standard(self) -> None:
        if self.match_dense_init:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self._weight_flat, -bound, bound)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            degree = torch.bincount(self.gene_idx, minlength=self.out_features).clamp(min=1)
            bound_e = degree[self.gene_idx].rsqrt()  # (n_edges,)
            with torch.no_grad():
                self._weight_flat.uniform_(-1.0, 1.0).mul_(bound_e)
            if self.bias is not None:
                bound_b = degree.rsqrt()  # (out_features,)
                tmp = torch.empty_like(self.bias).uniform_(-1.0, 1.0)
                with torch.inference_mode():
                    self.bias.copy_(tmp.mul_(bound_b))

    def _reset_pos(self) -> None:
        """Positive-weight init: ``U(0, 2*bound)`` (matches ``reset_params_pos``)."""
        if self.match_dense_init:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self._weight_flat, 0.0, 2.0 * bound)
            if self.bias is not None:
                nn.init.uniform_(self.bias, 0.0, 2.0 * bound)
        else:
            degree = torch.bincount(self.gene_idx, minlength=self.out_features).clamp(min=1)
            bound_e = degree[self.gene_idx].rsqrt()
            with torch.no_grad():
                self._weight_flat.uniform_(0.0, 1.0).mul_(2.0 * bound_e)
            if self.bias is not None:
                bound_b = degree.rsqrt()
                tmp = torch.empty_like(self.bias).uniform_(0.0, 1.0)
                with torch.inference_mode():
                    self.bias.copy_(tmp.mul_(2.0 * bound_b))

    def _reset_ensemble(self) -> None:
        """Init ``r``, ``s``, ``ensemble_bias`` (BatchEnsemble adapters)."""
        if self.k is None:
            return
        scaling_fn = {"ones": nn.init.ones_, "random-signs": _init_random_signs_}[self._scaling_init]
        if self.r is not None:
            scaling_fn(self.r)
        if self.s is not None:
            scaling_fn(self.s)
        if self.ensemble_bias is not None:
            bias_init = torch.empty(self.out_features, dtype=self._weight_flat.dtype, device=self._weight_flat.device)
            _init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.ensemble_bias.copy_(bias_init.unsqueeze(0).expand(self.k, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the sparse masked linear transform.

        Accepts inputs of shape ``(..., in_features)`` -- the feature axis is
        always the last dimension, matching ``nn.Linear``'s leading-dim
        broadcasting. Handles both 2D ``(B, in_features)`` and 3D
        ``(B, K, in_features)`` (the latter is required for TabM, which
        replicates the batch across the ``k`` ensemble dimension).

        Parameters
        ----------
        x : torch.Tensor, shape (..., in_features)

        Returns
        -------
        torch.Tensor, shape (..., out_features)
        """
        if self.r is not None:
            x = x * self.r
        contrib = self._weight_flat * x[..., self.cpg_idx]  # (..., n_edges)
        out = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        out.index_add_(-1, self.gene_idx, contrib)  # (..., out_features)
        if self.bias is not None:
            out = out + self.bias
        if self.s is not None:
            out = out * self.s
        if self.ensemble_bias is not None:
            out = out + self.ensemble_bias
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Dense ``(out_features, in_features)`` weight view -- mostly zeros.

        .. warning::
            At CpG scale this materializes a ~18 GB tensor. Use only for
            interpretability on CPU/small subsets. Prefer :meth:`weight_sparse`.
            The trainable parameter is the 1-D :attr:`_weight_flat`.
        """
        w = torch.zeros(
            self.out_features,
            self.in_features,
            device=self._weight_flat.device,
            dtype=self._weight_flat.dtype,
        )
        w[self.gene_idx, self.cpg_idx] = self._weight_flat
        return w

    def weight_sparse(self) -> torch.Tensor:
        """Memory-efficient sparse COO view of the weight.

        Returns
        -------
        torch.Tensor
            Coalesced sparse_coo_tensor of shape ``(out_features, in_features)``
            with ``nnz == n_edges`` at positions ``(gene_idx, cpg_idx)``.
        """
        idx = torch.stack([self.gene_idx, self.cpg_idx], dim=0)  # (2, n_edges)
        return torch.sparse_coo_tensor(idx, self._weight_flat, (self.out_features, self.in_features)).coalesce()

    def extra_repr(self) -> str:
        return (
            f"input_features={self.in_features}, output_features={self.out_features}, "
            f"n_edges={self.n_edges}, bias={self.bias is not None}, "
            f"match_dense_init={self.match_dense_init}, k={self.k}"
        )


class GatedSparseLinearLayer(nn.Module):
    """Gated (GLU-style) sparse masked linear over an arbitrary edge set.

    Two parallel sparse paths -- value and gate -- share the same
    (``in_idx``, ``out_idx``) edge structure; each path has its own flat 1-D
    weight over ``n_edges``. The output is the elementwise product of the
    value path and a nonlinearity applied to the gate path::

        value = scatter_add(_value_weight_flat * x[..., in_idx], out_idx) + value_bias
        gate  = scatter_add(_gate_weight_flat  * x[..., in_idx], out_idx) + gate_bias
        out   = value * gate_activation(gate)

    Designed as a general-purpose primitive: the mask may encode any
    bipartite or graph-structured sparsity pattern. A BINN-style stack
    instantiates one layer per stage with the appropriate ``mask_indices``
    artifact, e.g.::

        Layer 0: GatedSparseLinearLayer(cpg->gene mask,        in=n_cpg,      out=n_genes)
        Layer 1: GatedSparseLinearLayer(gene->pathway mask,    in=n_genes,    out=n_pathways)
        Layer 2: GatedSparseLinearLayer(pathway->pathway mask, in=n_pathways, out=n_pathways)

    Mathematically equivalent to two masked dense ``nn.Linear`` followed by an
    elementwise gate, where the edge structure (``in_idx``, ``out_idx``)
    *is* the mask -- no dense mask tensor is ever materialized in the
    forward/backward/optimizer hot path.

    Artifacts consumed (per fold/stage, exported by the
    ``Epigenetic-Dataset-Management`` repo)::

        mask_indices.npy   (2, n_edges) int64  -- row0=input_idx, row1=output_idx
        shapes.json        in_features, out_features, n_edges, ...

    Index convention: edge ``e`` corresponds to
    ``W[out_idx[e], in_idx[e]]`` of an ``nn.Linear(in_features, out_features)``
    weight of shape ``(out_features, in_features)``.

    Parameters
    ----------
    mask_indices : array_like, shape (2, n_edges), int64
        Edge index pairs. Row 0 = input feature index (column of ``x``),
        row 1 = output feature index (row of the equivalent dense weight).
        Accepted as a torch tensor, numpy array, or list-of-lists; coerced
        to ``torch.long``.
    in_features : int
        Number of input features. ``x.size(-1)`` must equal this.
    out_features : int
        Number of output features. ``out_idx`` values must be in
        ``[0, out_features)``.
    value_bias : bool, default True
        If True, adds a learnable per-output bias to the value path.
    gate_bias : bool, default False
        If True, adds a learnable per-output bias to the gate path (lets the
        gate learn a per-output baseline openness). Opt-in: default off so
        the minimal-param case is ``2 * n_edges + out_features``.
    match_dense_init : bool, default True
        If True, initialize both flat weights and value bias from
        ``U(-1/sqrt(in_features), 1/sqrt(in_features))`` -- distributionally
        identical to ``CustomizedLinear.__init__``'s
        ``kaiming_uniform_(a=sqrt(5))``. If False, use a sparse-aware
        per-output-degree init: each edge ``e`` feeding output ``o`` is
        drawn from ``U(-1/sqrt(degree(o)), 1/sqrt(degree(o)))``.
    pos_weights : bool, default False
        If True, override the chosen mode and sample both flat weights and
        value bias from ``U(0, 2*bound)`` (matches
        ``CustomizedLinear.reset_params_pos``).
    gate_activation : str, default "silu"
        Nonlinearity applied to the gate path before the elementwise
        product. One of ``"silu"`` (SwiGLU-style, ``x * silu(g)``) or
        ``"sigmoid"`` (classic GLU, ``x * sigmoid(g)``).
    gate_bias_init : float or None, default None
        If not None, override the gate bias init to this constant value.
        Set to a positive value (e.g. 3.0) so the gate starts "open":
        ``silu(3.0) ~ 2.96``, making the gated layer behave like a plain
        sparse layer at init. The gate can then learn to close/modulate.
        If None, the gate bias uses the same uniform init as the value
        path (``gate_bias=True`` must be set for this to take effect).

        For a **neutral** starting point (gate = 1.0 at init, the
        multiplicative identity), set ``gate_bias_init=1.2785`` with
        ``gate_weight_init="zero"`` (pure neutral) or
        ``gate_weight_init="scaled"`` (near-neutral with symmetry
        breaking).  ``silu(1.2785) = 1.0`` and ``silu'(1.2785) ~ 0.99``
        (the steepest point of the SiLU curve), so the gate can
        deviate in either direction—amplify or suppress—with maximum
        gradient sensitivity.
    gate_weight_init : str, default "default"
        Controls how ``_gate_weight_flat`` is initialized after the
        standard ``_init_flat_path`` call.  One of:

        * ``"default"`` — same random init as the value path
          (``U(-bound, bound)``).  This is the backward-compatible
          behavior.
        * ``"zero"`` — all gate weights set to zero, so the gate is
          purely determined by the bias: ``gate = silu(gate_bias)``
          for all inputs.  Combined with ``gate_bias_init=1.2785``
          this gives a pure neutral starting point (gate = 1.0).
        * ``"scaled"`` — gate weights are sampled from the standard
          uniform init then multiplied by ``gate_weight_scale``.
          Combined with ``gate_bias_init=1.2785`` this gives a
          near-neutral starting point with small symmetry-breaking
          noise.
    gate_weight_scale : float, default 0.01
        Multiplier applied to the uniform init when
        ``gate_weight_init="scaled"``.  At ``0.0`` equivalent to
        ``"zero"``, at ``1.0`` equivalent to ``"default"``.  Must be
        positive when ``gate_weight_init="scaled"``.
    k : int or None, default None
        If set, enables BatchEnsemble mode: per-member input scaling ``r``
        (``k, in_features``), output scaling ``s`` (``k, out_features``), and
        per-member ``ensemble_bias`` (``k, out_features``) are created. When
        None, the layer is a pure sparse primitive (no ensemble params). The
        ``r`` scaling is shared between value and gate paths (applied to the
        input before the split).
    ensemble_scaling_in : bool, default True
        If True (and ``k`` is set), create ``r``.
    ensemble_scaling_out : bool, default True
        If True (and ``k`` is set), create ``s``.
    ensemble_bias : bool, default True
        If True (and ``k`` is set), create per-member bias.
    scaling_init : str, default "ones"
        Init for ``r`` and ``s``: ``"ones"`` or ``"random-signs"``.

    Attributes
    ----------
    in_idx : torch.Tensor            (buffer, int64, shape (n_edges,))
    out_idx : torch.Tensor           (buffer, int64, shape (n_edges,))
    _value_weight_flat : nn.Parameter       (shape (n_edges,))
    _gate_weight_flat : nn.Parameter        (shape (n_edges,))
    value_bias : nn.Parameter or None       (shape (out_features,))
    gate_bias : nn.Parameter or None        (shape (out_features,))
    r : nn.Parameter or None          (shape (k, in_features))  -- ensemble only
    s : nn.Parameter or None          (shape (k, out_features)) -- ensemble only
    ensemble_bias : nn.Parameter or None (shape (k, out_features)) -- ensemble only

    Attributes (populated during forward)
    --------------------------------------
    _last_gate : torch.Tensor or None
        Detached gate activation ``gate_activation(g)`` from the most recent
        forward pass, shape ``(..., out_features)``. ``None`` before the first
        forward. Useful for interpretability analysis of per-sample gate
        modulation. Use :meth:`gate_weight_sparse` for weight inspection.
    _last_value : torch.Tensor or None
        Detached value path ``v`` from the most recent forward pass, same
        shape as ``_last_gate``. ``None`` before the first forward.

    Notes
    -----
    The trainable parameters are named ``_value_weight_flat`` and
    ``_gate_weight_flat`` (not ``weight``) to avoid clashing with the
    ``.value_weight`` / ``.gate_weight`` introspection shims (see those
    properties), which materialize dense ``(out_features, in_features)``
    tensors (~18 GB at CpG scale). Use :meth:`value_weight_sparse` and
    :meth:`gate_weight_sparse` for memory-efficient COO views.
    """

    __constants__ = [
        "in_features",
        "out_features",
        "n_edges",
        "gate_activation",
    ]

    in_idx: torch.Tensor
    out_idx: torch.Tensor
    _value_weight_flat: nn.Parameter
    _gate_weight_flat: nn.Parameter
    value_bias: nn.Parameter | None
    gate_bias: nn.Parameter | None
    r: nn.Parameter | None
    s: nn.Parameter | None
    ensemble_bias: nn.Parameter | None

    def __init__(
        self,
        mask_indices,
        in_features: int,
        out_features: int,
        value_bias: bool = True,
        gate_bias: bool = False,
        match_dense_init: bool = True,
        pos_weights: bool = False,
        gate_activation: str = "silu",
        k: int | None = None,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = True,
        scaling_init: Literal["ones", "random-signs"] = "ones",
        gate_bias_init: float | None = None,
        gate_weight_init: Literal["default", "zero", "scaled"] = "default",
        gate_weight_scale: float = 0.01,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if gate_weight_init not in ("default", "zero", "scaled"):
            raise ValueError(f"gate_weight_init must be 'default', 'zero', or 'scaled', got {gate_weight_init!r}")
        if gate_weight_init == "scaled" and gate_weight_scale <= 0:
            raise ValueError(
                f"gate_weight_scale must be positive when gate_weight_init='scaled', got {gate_weight_scale}"
            )
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if gate_activation not in ("silu", "sigmoid"):
            raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")

        idx = torch.as_tensor(mask_indices, dtype=torch.long)
        if idx.ndim != 2 or idx.size(0) != 2:
            raise ValueError(f"mask_indices must have shape (2, n_edges), got {tuple(idx.shape)}")
        n_edges = idx.size(1)

        in_idx = idx[0]
        out_idx = idx[1]
        if in_idx.size(0) != out_idx.size(0):
            raise ValueError("mask_indices rows must have equal length")
        if n_edges > 0:
            if int(in_idx.max().item()) >= in_features:
                raise ValueError("in_idx contains an index >= in_features")
            if int(out_idx.max().item()) >= out_features:
                raise ValueError("out_idx contains an index >= out_features")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_edges = int(n_edges)
        self.match_dense_init = bool(match_dense_init)
        self.pos_weights = bool(pos_weights)
        self.gate_activation = gate_activation
        self.k = k
        self.gate_bias_init = gate_bias_init
        self.gate_weight_init = gate_weight_init
        self.gate_weight_scale = float(gate_weight_scale)

        self.register_buffer("in_idx", in_idx)
        self.register_buffer("out_idx", out_idx)

        self._value_weight_flat = nn.Parameter(torch.empty(n_edges))
        self._gate_weight_flat = nn.Parameter(torch.empty(n_edges))
        if value_bias:
            self.value_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("value_bias", None)
        if gate_bias:
            self.gate_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("gate_bias", None)

        self.r: nn.Parameter | None = None
        self.s: nn.Parameter | None = None
        self.ensemble_bias: nn.Parameter | None = None
        if k is not None:
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            if ensemble_scaling_in:
                self.r = nn.Parameter(torch.empty(k, in_features))
            if ensemble_scaling_out:
                self.s = nn.Parameter(torch.empty(k, out_features))
            if ensemble_bias:
                self.ensemble_bias = nn.Parameter(torch.empty(k, out_features))
        self._scaling_init = scaling_init

        self._last_gate: torch.Tensor | None = None
        self._last_value: torch.Tensor | None = None
        self._last_gate_grad: torch.Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-sample both flat weights (and value/gate biases) from a uniform dist.

        With ``match_dense_init=True`` the bound is ``1/sqrt(in_features)``,
        matching ``CustomizedLinear.__init__``'s ``kaiming_uniform_(a=sqrt(5))``.
        With ``match_dense_init=False`` each edge uses ``1/sqrt(degree(output))``.
        ``pos_weights=True`` shifts the range to ``U(0, 2*bound)``. Applied
        independently to the value and gate paths. Ensemble adapters (``r``,
        ``s``, ``ensemble_bias``) are initialized when ``k`` is set.

        After the standard init, the gate weight is optionally overridden
        based on ``gate_weight_init``: ``"zero"`` sets all gate weights to
        zero, ``"scaled"`` multiplies them by ``gate_weight_scale``.
        """
        self._init_flat_path(self._value_weight_flat, self.value_bias)
        self._init_flat_path(self._gate_weight_flat, self.gate_bias)
        if self.gate_weight_init == "zero":
            nn.init.zeros_(self._gate_weight_flat)
        elif self.gate_weight_init == "scaled":
            self._gate_weight_flat.data.mul_(self.gate_weight_scale)
        if self.gate_bias is not None and self.gate_bias_init is not None:
            nn.init.constant_(self.gate_bias, self.gate_bias_init)
        self._reset_ensemble()

    def _reset_ensemble(self) -> None:
        """Init ``r``, ``s``, ``ensemble_bias`` (BatchEnsemble adapters)."""
        if self.k is None:
            return
        scaling_fn = {"ones": nn.init.ones_, "random-signs": _init_random_signs_}[self._scaling_init]
        if self.r is not None:
            scaling_fn(self.r)
        if self.s is not None:
            scaling_fn(self.s)
        if self.ensemble_bias is not None:
            bias_init = torch.empty(
                self.out_features, dtype=self._value_weight_flat.dtype, device=self._value_weight_flat.device
            )
            _init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.ensemble_bias.copy_(bias_init.unsqueeze(0).expand(self.k, -1))

    def _init_flat_path(self, weight_flat: nn.Parameter, bias: nn.Parameter | None) -> None:
        """Initialize one flat weight vector and its optional bias."""
        if self.pos_weights:
            if self.match_dense_init:
                bound = 1.0 / math.sqrt(self.in_features)
                nn.init.uniform_(weight_flat, 0.0, 2.0 * bound)
                if bias is not None:
                    nn.init.uniform_(bias, 0.0, 2.0 * bound)
            else:
                degree = torch.bincount(self.out_idx, minlength=self.out_features).clamp(min=1)
                bound_e = degree[self.out_idx].rsqrt()  # (n_edges,)
                with torch.no_grad():
                    weight_flat.uniform_(0.0, 1.0).mul_(2.0 * bound_e)
                if bias is not None:
                    bound_b = degree.rsqrt()  # (out_features,)
                    tmp = torch.empty_like(bias).uniform_(0.0, 1.0)
                    with torch.inference_mode():
                        bias.copy_(tmp.mul_(2.0 * bound_b))
            return

        if self.match_dense_init:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(weight_flat, -bound, bound)
            if bias is not None:
                nn.init.uniform_(bias, -bound, bound)
        else:
            degree = torch.bincount(self.out_idx, minlength=self.out_features).clamp(min=1)
            bound_e = degree[self.out_idx].rsqrt()  # (n_edges,)
            with torch.no_grad():
                weight_flat.uniform_(-1.0, 1.0).mul_(bound_e)
            if bias is not None:
                bound_b = degree.rsqrt()  # (out_features,)
                tmp = torch.empty_like(bias).uniform_(-1.0, 1.0)
                with torch.inference_mode():
                    bias.copy_(tmp.mul_(bound_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated sparse masked linear transform.

        Accepts inputs of shape ``(..., in_features)`` -- the feature axis is
        always the last dimension, matching ``nn.Linear``'s leading-dim
        broadcasting. Handles both 2D ``(B, in_features)`` and 3D
        ``(B, K, in_features)`` (the latter is required for TabM, which
        replicates the batch across the ``k`` ensemble dimension).

        Parameters
        ----------
        x : torch.Tensor, shape (..., in_features)

        Returns
        -------
        torch.Tensor, shape (..., out_features)
        """
        if self.r is not None:
            x = x * self.r
        v_contrib = self._value_weight_flat * x[..., self.in_idx]  # (..., n_edges)
        g_contrib = self._gate_weight_flat * x[..., self.in_idx]  # (..., n_edges)
        v = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        g = torch.zeros_like(v)
        v.index_add_(-1, self.out_idx, v_contrib)  # (..., out_features)
        g.index_add_(-1, self.out_idx, g_contrib)  # (..., out_features)
        if self.value_bias is not None:
            v = v + self.value_bias
        if self.gate_bias is not None:
            g = g + self.gate_bias
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(g)
        else:  # "silu"
            gate = F.silu(g)
        self._last_gate = gate.detach()
        self._last_value = v.detach()
        self._last_gate_grad = gate
        out = v * gate
        if self.s is not None:
            out = out * self.s
        if self.ensemble_bias is not None:
            out = out + self.ensemble_bias
        return out

    @property
    def value_weight(self) -> torch.Tensor:
        """Dense ``(out_features, in_features)`` value-weight view -- mostly zeros.

        .. warning::
            At CpG scale this materializes a ~18 GB tensor. Use only for
            interpretability on CPU/small subsets. Prefer
            :meth:`value_weight_sparse`. The trainable parameter is the 1-D
            :attr:`_value_weight_flat`.
        """
        w = torch.zeros(
            self.out_features,
            self.in_features,
            device=self._value_weight_flat.device,
            dtype=self._value_weight_flat.dtype,
        )
        w[self.out_idx, self.in_idx] = self._value_weight_flat
        return w

    @property
    def gate_weight(self) -> torch.Tensor:
        """Dense ``(out_features, in_features)`` gate-weight view -- mostly zeros.

        .. warning::
            At CpG scale this materializes a ~18 GB tensor. Use only for
            interpretability on CPU/small subsets. Prefer
            :meth:`gate_weight_sparse`. The trainable parameter is the 1-D
            :attr:`_gate_weight_flat`.
        """
        w = torch.zeros(
            self.out_features,
            self.in_features,
            device=self._gate_weight_flat.device,
            dtype=self._gate_weight_flat.dtype,
        )
        w[self.out_idx, self.in_idx] = self._gate_weight_flat
        return w

    def value_weight_sparse(self) -> torch.Tensor:
        """Memory-efficient sparse COO view of the value weight.

        Returns
        -------
        torch.Tensor
            Coalesced sparse_coo_tensor of shape ``(out_features, in_features)``
            with ``nnz == n_edges`` at positions ``(out_idx, in_idx)``.
        """
        idx = torch.stack([self.out_idx, self.in_idx], dim=0)  # (2, n_edges)
        return torch.sparse_coo_tensor(idx, self._value_weight_flat, (self.out_features, self.in_features)).coalesce()

    def gate_weight_sparse(self) -> torch.Tensor:
        """Memory-efficient sparse COO view of the gate weight.

        Returns
        -------
        torch.Tensor
            Coalesced sparse_coo_tensor of shape ``(out_features, in_features)``
            with ``nnz == n_edges`` at positions ``(out_idx, in_idx)``.
        """
        idx = torch.stack([self.out_idx, self.in_idx], dim=0)  # (2, n_edges)
        return torch.sparse_coo_tensor(idx, self._gate_weight_flat, (self.out_features, self.in_features)).coalesce()

    def extra_repr(self) -> str:
        return (
            f"input_features={self.in_features}, output_features={self.out_features}, "
            f"n_edges={self.n_edges}, value_bias={self.value_bias is not None}, "
            f"gate_bias={self.gate_bias is not None}, "
            f"match_dense_init={self.match_dense_init}, "
            f"gate_activation={self.gate_activation}, k={self.k}, "
            f"gate_weight_init={self.gate_weight_init}"
        )
