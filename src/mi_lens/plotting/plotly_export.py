from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image


def _to_image_bytes(fig: go.Figure, *, format: str, scale: int = 2) -> bytes:
    return pio.to_image(fig, format=format, engine="kaleido", scale=scale)


def _write_plotly_pdf(fig: go.Figure, output_path: Path) -> Path:
    try:
        pdf_bytes = _to_image_bytes(fig, format="pdf", scale=2)
        output_path.write_bytes(pdf_bytes)
        return output_path
    except Exception:
        png_bytes = _to_image_bytes(fig, format="png", scale=2)
        png_path = output_path.with_suffix(".png")
        png_path.write_bytes(png_bytes)
        image = Image.open(png_path).convert("RGB")
        image.save(output_path, "PDF", resolution=300.0)
        return output_path


def save_plotly_figure(
    fig: go.Figure,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = (format or output_path.suffix.lstrip(".") or "pdf").lower()
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{resolved}")

    if resolved == "html":
        pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
        return output_path
    if resolved in {"png", "svg", "jpeg", "jpg", "webp"}:
        image_format = "jpeg" if resolved == "jpg" else resolved
        image_bytes = _to_image_bytes(fig, format=image_format, scale=2)
        output_path.write_bytes(image_bytes)
        return output_path
    if resolved == "pdf":
        return _write_plotly_pdf(fig, output_path)
    raise ValueError(f"Unsupported Plotly export format: {resolved!r} for {output_path}")