"""Notebook plotting helpers for JL vs. logit-lens comparisons."""

from .lens_diff_widget import (
    build_lens_comparison_widget,
    show_lens_comparison_widget,
)

build_lens_diff_widget = build_lens_comparison_widget
show_lens_diff_widget = show_lens_comparison_widget

__all__ = [
    "build_lens_comparison_widget",
    "show_lens_comparison_widget",
    "build_lens_diff_widget",
    "show_lens_diff_widget",
]
