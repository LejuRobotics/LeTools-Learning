import math
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from ..parallel_state import get_parallel_state
from ...utils.logging import get_logger


logger = get_logger(__name__)


def ep_fsdp2_clip_grad_norm(
    model,
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool = True,
    parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """EP-aware gradient clipping for FSDP2.

    Separates EP params and non-EP params, reduces their norms over the
    correct process groups, then applies a single global clip coefficient.

    - non-EP params: local norm -> all-reduce over fsdp_group
    - EP params: local norm -> all-reduce over ep_fsdp_group -> all-reduce over ep_group

    If ``parameters`` is provided, clips only that subset. ``None`` keeps the
    previous full-model behaviour.
    """
    ps = get_parallel_state()
    ep_param_set = model._ep_param_set

    if parameters is None:
        params = list(model.parameters())
    else:
        params = list(parameters)
    params = [p for p in params if p.grad is not None]

    ep_params = [p for p in params if p in ep_param_set]
    non_ep_params = [p for p in params if p not in ep_param_set]

    if math.isinf(norm_type):
        # inf-norm: take elementwise MAX
        non_ep_norm = _local_max(non_ep_params)
        dist.all_reduce(non_ep_norm, op=dist.ReduceOp.MAX, group=ps.fsdp_group)

        ep_norm = _local_max(ep_params)
        ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group()
        dist.all_reduce(ep_norm, op=dist.ReduceOp.MAX, group=ep_fsdp_group)
        dist.all_reduce(ep_norm, op=dist.ReduceOp.MAX, group=ps.ep_group)

        total_norm = torch.maximum(non_ep_norm, ep_norm)
    else:
        p = float(norm_type)

        # non-EP: reduce over fsdp
        non_ep_norm = _local_pth_sum(non_ep_params, p)
        dist.all_reduce(non_ep_norm, op=dist.ReduceOp.SUM, group=ps.fsdp_group)

        # EP: reduce over ep_fsdp, then ep
        ep_norm = _local_pth_sum(ep_params, p)
        ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group()
        dist.all_reduce(ep_norm, op=dist.ReduceOp.SUM, group=ep_fsdp_group)
        dist.all_reduce(ep_norm, op=dist.ReduceOp.SUM, group=ps.ep_group)

        total_norm = (non_ep_norm + ep_norm) ** (1.0 / p)

    # Apply single global clip coefficient.
    # Must clip EP and non-EP params separately because their grads are
    # DTensors on different meshes (ep_fsdp vs dp_shard), and PyTorch's
    # clip_grads_with_norm_ does pointwise ops that reject mesh mismatches.
    if non_ep_params:
        torch.nn.utils.clip_grads_with_norm_(
            non_ep_params, max_norm, total_norm, foreach=foreach
        )
    if ep_params:
        torch.nn.utils.clip_grads_with_norm_(
            ep_params, max_norm, total_norm, foreach=foreach
        )

    return total_norm


def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    """Compute local sum of p-th powers of gradient norms."""
    grads = [param.grad for param in params if param.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor)
        else g.detach().to(torch.float32)
        for g in grads
    ]

    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device("cuda")
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)

    with torch.no_grad():
        grouped = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads], _) in grouped.items():
            if _has_foreach_support(device_grads, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads, p), p)
                res += torch.sum(torch.stack(out)).to(default_device)
            else:
                for grad in device_grads:
                    gn = torch.norm(grad, p=p)
                    res = res + (gn ** p).to(default_device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Compute local max of absolute gradient values."""
    dev = None
    mx = None
    for param in params:
        g = param.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device("cuda")
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx
