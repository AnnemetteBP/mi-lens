"""Method helpers for lens analysis."""

from .batch_capture import (
    BatchCaptureConfig,
    capture_lens_batch,
    compare_group_means,
    summarize_capture_rows,
    write_capture_rows_jsonl,
)
from .compatibility import (
    CompatibilityCheck,
    CompatibilityReport,
    build_compatibility_report,
)
from .data_prep import (
    MKQAExportSpec,
    PreparedDatasetManifest,
    ProjectDataPaths,
    TextExportSpec,
    export_mkqa_to_jsonl,
    export_text_split_to_jsonl,
    project_data_paths,
    write_dataset_registry,
)
from .fuzzy_trace import (
    FuzzyTraceConfig,
    LensTokenMetrics,
    evaluate_token_metrics,
)
from .fragmentation import *
from .lens_eval import (
    LensEvalConfig,
    LensSurfaceMetrics,
    ResidualAlignmentMetrics,
    SurfaceFormConfig,
    evaluate_residual_alignment,
    evaluate_surface_metrics,
)
from .routing_eval import (
    summarize_router_coactivation,
    summarize_router_probabilities,
    summarize_router_score_coactivation,
    summarize_router_scores,
)
from .tokenization_audit import (
    TokenAuditConfig,
    TokenAuditRow,
    build_token_audit_rows,
    pair_token_audit_rows,
    summarize_paired_token_audit_rows,
    summarize_token_audit_rows,
)

__all__ = [
    "BatchCaptureConfig",
    "capture_lens_batch",
    "compare_group_means",
    "summarize_capture_rows",
    "write_capture_rows_jsonl",
    "CompatibilityCheck",
    "CompatibilityReport",
    "build_compatibility_report",
    "MKQAExportSpec",
    "PreparedDatasetManifest",
    "ProjectDataPaths",
    "TextExportSpec",
    "export_mkqa_to_jsonl",
    "export_text_split_to_jsonl",
    "project_data_paths",
    "write_dataset_registry",
    "FuzzyTraceConfig",
    "LensTokenMetrics",
    "evaluate_token_metrics",
    "LensEvalConfig",
    "LensSurfaceMetrics",
    "ResidualAlignmentMetrics",
    "SurfaceFormConfig",
    "evaluate_surface_metrics",
    "evaluate_residual_alignment",
    "TokenAuditConfig",
    "TokenAuditRow",
    "build_token_audit_rows",
    "pair_token_audit_rows",
    "summarize_paired_token_audit_rows",
    "summarize_token_audit_rows",
    "summarize_router_probabilities",
    "summarize_router_scores",
    "summarize_router_coactivation",
    "summarize_router_score_coactivation",
]
