"""Sparse-feature analysis namespace.

These modules are placeholders for scalable feature methods such as ITDA, SAEs,
crosscoders, and RouterInterp-style routing explanations.
"""

__all__ = [
    "itda",
    "sae",
    "crosscoder",
    "router_interp",
    "feature_metrics",
    "feature_capture",
    "feature_alignment",
    "ResidualCaptureConfig",
    "capture_residual_activation_shards",
    "RouterLayerCapture",
    "RouterInterpCaptureConfig",
    "RoutingProbe",
    "capture_flexolmo_router_layers",
    "capture_routerinterp_prompt_artifacts",
    "feature_activation_diagnostics",
    "feature_coactivation_ratio",
    "fit_routing_probe",
    "macro_f1_expert_routing",
    "rho_usefulness",
    "routing_recall",
    "selected_expert_targets",
    "sparse_by_magnitude",
    "top_k_expert_predictions",
    "top_rho_features",
    "TopKSAE",
    "TopKSAEConfig",
    "fit_topk_sae",
    "ITDA",
    "ITDAConfig",
]

from .feature_capture import ResidualCaptureConfig, capture_residual_activation_shards
from .router_interp import (
    RouterLayerCapture,
    RouterInterpCaptureConfig,
    RoutingProbe,
    capture_flexolmo_router_layers,
    capture_routerinterp_prompt_artifacts,
    feature_activation_diagnostics,
    feature_coactivation_ratio,
    fit_routing_probe,
    macro_f1_expert_routing,
    rho_usefulness,
    routing_recall,
    selected_expert_targets,
    sparse_by_magnitude,
    top_k_expert_predictions,
    top_rho_features,
)
from .sae import TopKSAE, TopKSAEConfig, fit_topk_sae
from .itda import ITDA, ITDAConfig
