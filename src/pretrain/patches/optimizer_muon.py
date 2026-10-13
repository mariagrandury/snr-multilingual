# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon optimizer, ported from upstream NVIDIA Megatron-LM into the swiss-ai fork.

Sources (Apache-2.0):
  * Megatron-LM ``megatron/core/optimizer/muon.py`` @ core_v0.16.0 (the standalone
    ``get_megatron_muon_optimizer`` factory: Muon for 2D hidden weights, Adam for the rest)
    and ``megatron/core/optimizer/emerging_optimizers.py`` @ main 54f18cd
    (``TensorParallelMuon`` kwarg names/defaults).
  * NVIDIA-NeMo/Emerging-Optimizers v0.3.0 (b309e2f), which upstream Megatron imports:
    ``OrthogonalizedOptimizer``, ``WeightDecayMixin``, ``newton_schulz``,
    ``newton_schulz_tp``, ``get_muon_scale_factor``, ``fp32_matmul_precision``.
    Vendored here because the ``emerging_optimizers`` package is not in our training
    image (ngc-nemo 25.11.01). Dropped: the Triton SYRK kernel, batched (3D) NS, the
    "custom" coefficient set and absl logging.

Fork adaptations: no ``ProcessGroupCollection`` (TP groups come from ``mpu``), the
fork's ``get_megatron_optimizer`` signature, no layer-wise distributed optimizer
(``--use-distributed-optimizer`` is rejected for muon, as in core_v0.16.0).
"""

import logging
from contextlib import contextmanager
from itertools import chain, cycle, islice, repeat
from typing import Any, Callable, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core import mpu

from ..transformer.module import MegatronModule
from ..utils import log_single_rank
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# Vendored from Emerging-Optimizers v0.3.0
# ===========================================================================

_COEFFICIENT_SETS = {
    "simple": [(3.4445, -4.7750, 2.0315)],
    "quintic": [
        # https://leloykun.github.io/ponder/muon-opt-coeffs/ (modded-nanogpt values)
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # https://arxiv.org/abs/2505.16932 (includes the 1.01^degree safety division)
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "cans": [
        # http://arxiv.org/abs/2506.10935
        (8.4703, -25.1081, 18.6293),
        (4.1828, -3.1087, 0.5806),
        (3.9619, -2.9541, 0.5630),
        (3.2866, -2.4647, 0.5074),
        (2.2737, -1.6447, 0.4162),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
    "deepseekv4": [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2,
}
_REPEAT_LAST_TYPES = ("polar_express", "cans", "deepseekv4")
MUON_COEFFICIENT_TYPES = tuple(_COEFFICIENT_SETS)


@contextmanager
def fp32_matmul_precision(precision: str = "highest"):
    """Temporarily set torch's float32 matmul precision."""
    prev_val = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_val)


def _distributed_normalize_p2(x, eps, group):
    x_sq_sum = (x * x).sum()
    torch.distributed.all_reduce(x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=group)
    return x / torch.sqrt(x_sq_sum).clamp_min(eps)


def _newton_schulz_step(X, a, b, c, tp_group=None):
    A = X @ X.mT
    if tp_group is not None:
        torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    return torch.addmm(X, B, X, alpha=1.0, beta=a)


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "quintic",
    eps: float = 1e-7,
    transpose: Optional[bool] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Newton-Schulz approximation of the orthogonal polar factor of a 2D fp32 tensor."""
    if x.ndim != 2:
        raise TypeError(f"Input tensor x must be 2d, got {x.ndim}d")
    if x.dtype != torch.float32:
        raise TypeError(f"Input tensor x must be in float32, got {x.dtype}")
    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    # whiten along the smaller dimension
    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # spectral norm <= Frobenius norm <= 1
    if tp_group is not None:
        X = _distributed_normalize_p2(x, eps, tp_group)
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    if coefficient_type in _REPEAT_LAST_TYPES:
        base = chain(coefficient_sets, repeat(coefficient_sets[-1]))
    else:
        base = cycle(coefficient_sets)

    if torch.get_float32_matmul_precision() == "medium":
        # No FP32-I/O BF16-compute kernels in PyTorch: run the iteration in bf16.
        X = X.to(torch.bfloat16)

    for a, b, c in islice(base, steps):
        X = _newton_schulz_step(X, a, b, c, tp_group=tp_group)

    X = X.to(torch.float32)
    if transpose:
        X = X.mT
    return X


def newton_schulz_tp(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group: Optional[torch.distributed.ProcessGroup],
    partition_dim: Optional[int] = None,
    tp_mode: Literal["duplicated", "distributed"] = "duplicated",
) -> torch.Tensor:
    """Tensor-parallel Newton-Schulz. partition_dim=None means "not TP-sharded"."""
    if partition_dim is None or tp_group is None or tp_group.size() == 1:
        return newton_schulz(x, steps, coefficient_type)

    if tp_mode == "duplicated":
        x_shards = [torch.empty_like(x) for _ in range(tp_group.size())]
        torch.distributed.all_gather(x_shards, x, tp_group)
        global_x = torch.cat(x_shards, dim=partition_dim)
        orthogonalized_x = newton_schulz(global_x, steps, coefficient_type)
        return orthogonalized_x.chunk(tp_group.size(), dim=partition_dim)[tp_group.rank()]
    if tp_mode == "distributed":
        if partition_dim not in (0, 1):
            raise ValueError(f"Invalid partition_dim: {partition_dim}")
        return newton_schulz(
            x, steps, coefficient_type, transpose=(partition_dim == 0), tp_group=tp_group
        )
    raise ValueError(f"Invalid tp_mode: {tp_mode}")


def get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Scale applied to the orthogonalized update."""
    if mode == "shape_scaling":
        # https://kellerjordan.github.io/posts/muon/
        return max(1, size_out / size_in) ** 0.5
    if mode == "spectral":
        # https://arxiv.org/abs/2502.16982
        return max(size_out, size_in) ** 0.5
    if mode == "unit_rms_norm":
        # https://arxiv.org/abs/2502.07529
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")


MUON_SCALE_MODES = ("shape_scaling", "spectral", "unit_rms_norm")


class OrthogonalizedOptimizer(torch.optim.Optimizer):
    """Momentum SGD whose update is orthogonalized (Emerging-Optimizers base class)."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float,
        weight_decay: float,
        *,
        nesterov: bool,
        weight_decay_method: str,
        fp32_matmul_prec: str,
        scaled_orthogonalize_fn: Optional[Callable] = None,
    ):
        if scaled_orthogonalize_fn is None:
            scaled_orthogonalize_fn = torch.nn.Identity()
        self.fp32_matmul_prec = fp32_matmul_prec
        self.nesterov = nesterov
        self.weight_decay_method = weight_decay_method
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn

    def _apply_weight_decay_inplace(self, p, grad, lr, weight_decay):
        if weight_decay == 0.0:
            return
        if self.weight_decay_method == "decoupled":
            p.add_(p, alpha=(-weight_decay * lr))
        elif self.weight_decay_method == "independent":
            p.add_(p, alpha=-weight_decay)
        elif self.weight_decay_method == "l2":
            grad.add_(p, alpha=weight_decay)
        elif self.weight_decay_method == "palm":
            p.add_(p, alpha=(-(weight_decay * lr * lr)))
        else:
            raise ValueError(f"Invalid weight decay method: {self.weight_decay_method}")

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("closure is not supported")
        for group in self.param_groups:
            self._init_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
                # EMA-style momentum
                state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                if self.nesterov:
                    grad = grad.lerp(state["momentum_buffer"], group["momentum"])
                else:
                    grad = state["momentum_buffer"]
                with fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    orth_grad = self.orthogonalize(p, grad, **group_kwargs)
                p.add_(orth_grad, alpha=-group["lr"])
        return None

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if grad.ndim != 2:
            raise ValueError("Only 2D parameters are supported.")
        return self.scaled_orthogonalize_fn(grad)


# ===========================================================================
# Megatron Muon (upstream TensorParallelMuon, minus GTP / auto tp_mode / SYRK)
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[List[int]] = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if coefficient_type not in _COEFFICIENT_SETS:
            raise ValueError(
                f"Unsupported muon coefficient type '{coefficient_type}'. "
                f"Supported types: {MUON_COEFFICIENT_TYPES}"
            )

        def scaled_orthogonalize_fn(grad, tp_group, partition_dim=None):
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None and tp_group is not None:
                size[partition_dim] *= tp_group.size()
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method="decoupled" if use_decoupled_weight_decay else "l2",
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum ``grad`` of parameter ``p`` (QKV split per head group)."""
        if getattr(p, 'expert_tp', False):
            tp_group = mpu.get_expert_tensor_parallel_group(check_initialized=False)
        else:
            tp_group = mpu.get_tensor_model_parallel_group(check_initialized=False)
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if not (self.split_qkv and self.is_qkv_fn(p)):
            return self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)

        qkv_rows = sum(self.qkv_split_shapes)
        if grad.size(0) % qkv_rows != 0:
            raise RuntimeError(
                f"Muon QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                f"split_shapes={self.qkv_split_shapes}"
            )
        num_query_groups = grad.size(0) // qkv_rows
        cols = grad.size(-1)
        qkv_grads = torch.split(
            grad.view(num_query_groups, qkv_rows, cols), self.qkv_split_shapes, dim=1
        )
        qkv_grads = [
            self.scaled_orthogonalize_fn(g.reshape(-1, cols), tp_group, partition_dim).view(
                num_query_groups, -1, cols
            )
            for g in qkv_grads
        ]
        return torch.cat(qkv_grads, dim=1).view(grad.shape)


def _muon_init_state_fn(opt, config=None):
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)


def _wrap_muon(optimizer, config):
    if config.bf16:
        return Float16OptimizerWithFloat16Params(optimizer, config, None, _muon_init_state_fn)
    return FP32Optimizer(optimizer, config, _muon_init_state_fn)


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """Muon for 2D non-embedding weights, ``config.muon_scalar_optimizer`` for the rest.

    Same split as upstream: embeddings / output layer and every non-2D parameter (norms,
    biases, xIELU alphas) go to the scalar optimizer, built by the regular
    ``get_megatron_optimizer``. All parts share one ``config`` (and so one LR schedule,
    weight decay and global grad-norm clipping via ``ChainedOptimizer``).
    """
    # Imported here to avoid a circular import (this module is imported by __init__).
    from . import _get_param_groups, get_megatron_optimizer

    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')
    # The scalar optimizer is built through the normal path, which dispatches on
    # config.optimizer (upstream does the same with 'adam').
    config.optimizer = config.muon_scalar_optimizer

    log_single_rank(logger, logging.INFO, f'Setting up muon optimizer with config {config}')

    linear_params = []
    nonlinear_params = []
    qkv_split_shapes = None
    for model_chunk in model_chunks:
        # no need to check tp: tp splits by head and this is per head(group) dimension
        tcfg = model_chunk.config
        qkv_split_shapes = [
            tcfg.num_attention_heads // tcfg.num_query_groups * tcfg.kv_channels,
            tcfg.kv_channels,
            tcfg.kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    muon_kwargs = dict(
        lr=config.lr,
        momentum=config.muon_momentum,
        nesterov=config.muon_nesterov,
        weight_decay=config.weight_decay,
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        coefficient_type=config.muon_coefficient_type,
        num_ns_steps=config.muon_num_ns_steps,
        scale_mode=config.muon_scale_mode,
        split_qkv=config.muon_split_qkv,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        extra_scale_factor=config.muon_extra_scale_factor,
        tp_mode=config.muon_tp_mode,
    )

    # Freeze the scalar params so _get_param_groups only sees Muon's.
    for param in nonlinear_params:
        param.requires_grad = False
    linear_param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )
    expert_param_groups = [g for g in linear_param_groups if g['is_expert_parallel']]
    dense_param_groups = [g for g in linear_param_groups if not g['is_expert_parallel']]

    optimizers = []
    if dense_param_groups:
        optimizers.append(_wrap_muon(TensorParallelMuon(dense_param_groups, **muon_kwargs), config))
    if expert_param_groups:
        expert_optimizer = _wrap_muon(TensorParallelMuon(expert_param_groups, **muon_kwargs), config)
        setattr(
            expert_optimizer,
            'grad_stats_parallel_group',
            mpu.get_expert_tensor_model_pipeline_parallel_group(),
        )
        optimizers.append(expert_optimizer)

    # Swap: unfreeze scalar params, freeze Muon params, build the scalar optimizer.
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False
    try:
        scalar_optimizer = get_megatron_optimizer(
            config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult
        )
    finally:
        for param in linear_params:
            param.requires_grad = True

    if isinstance(scalar_optimizer, ChainedOptimizer):
        optimizers += scalar_optimizer.chained_optimizers
    else:
        optimizers.append(scalar_optimizer)
    return ChainedOptimizer(optimizers)
