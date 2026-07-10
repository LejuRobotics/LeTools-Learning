"""MoE statistics and monitoring utilities."""

import logging
# Ensure info_rank0 is monkey-patched onto logging.Logger
import lingbotvla.utils.logging  # noqa: F401

logger = logging.getLogger(__name__)


def log_model_param_stats(model):
    """Log VLM, action expert, and activated parameter counts.

    For MoE models, activated params = total - routed_params * (1 - top_k/num_experts).
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        Qwen2TokenMoeBlock, Qwen2FusedExperts,
    )

    def _numel(module):
        return sum(p.numel() for p in module.parameters())

    vlm_params = _numel(model.model.qwenvl_with_expert.qwenvl)
    action_expert_params = _numel(model.model.qwenvl_with_expert.qwen_expert)
    total_params = _numel(model)

    # MoE: only top_k / num_experts routed experts are activated per token
    activated_action_expert_params = action_expert_params
    routed_expert_params = 0
    moe_top_k = None
    moe_num_experts = None
    for m in model.model.qwenvl_with_expert.qwen_expert.modules():
        if isinstance(m, Qwen2TokenMoeBlock):
            moe_top_k = m.top_k
            moe_num_experts = m.num_experts
            routed_expert_params += _numel(m.experts)

    if routed_expert_params > 0 and moe_top_k is not None:
        activated_action_expert_params = (
            action_expert_params
            - routed_expert_params
            + routed_expert_params * moe_top_k // moe_num_experts
        )

    activated_total = total_params - action_expert_params + activated_action_expert_params
    moe_info = (
        f"  (top_k={moe_top_k}/{moe_num_experts} experts)"
        if routed_expert_params > 0 else "  (no MoE)"
    )

    # Action-side modules outside qwen_expert
    flow = model.model  # FlowMatching
    qwe = model.model.qwenvl_with_expert  # QwenvlWithExpertModel
    action_side_modules = {}
    for name in ("state_proj", "action_in_proj", "action_out_proj",
                 "action_time_mlp_in", "action_time_mlp_out",
                 "time_mlp_in", "time_mlp_out"):
        if hasattr(flow, name):
            action_side_modules[name] = _numel(getattr(flow, name))
    for name in ("expert_visual", "expert_visual_mlp"):
        if hasattr(qwe, name):
            action_side_modules[name] = _numel(getattr(qwe, name))
    action_side_total = sum(action_side_modules.values())

    # Format output
    sep = "=" * 60
    lines = [
        f"\n{sep}",
        "Model Parameter Statistics:",
        f"  VLM params:                    {vlm_params / 1e6:>10.1f}M",
        f"  Action expert params (total):  {action_expert_params / 1e6:>10.1f}M",
        f"  Action expert params (active): {activated_action_expert_params / 1e6:>10.1f}M{moe_info}",
    ]
    if action_side_modules:
        lines.append(f"  Action side modules (other):   {action_side_total / 1e6:>10.1f}M")
        for name, count in action_side_modules.items():
            lines.append(f"    - {name:30s} {count / 1e6:>8.2f}M")
    lines.extend([
        f"  Total params:                  {total_params / 1e6:>10.1f}M",
        f"  Activated total params:        {activated_total / 1e6:>10.1f}M",
        sep,
    ])
    logger.info_rank0("\n".join(lines))
