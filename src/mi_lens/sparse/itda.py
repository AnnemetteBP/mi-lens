"""Numerically checked ITDA sparse dictionaries for RouterInterp controls.

The implementation follows the vendored ``saes/itda/train.py`` algorithm:
matching-pursuit coefficients, normalized reconstruction-error gating, and a
dictionary made from observed activation vectors.  It is kept dependency-light
because the vendored training CLI imports W&B and Socket.IO before exposing
its core classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ITDAConfig:
    d_model: int
    max_atoms: int
    k: int
    loss_threshold: float


def _require_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")


class ITDA(nn.Module):
    """Inference-time dictionary of observed activation directions.

    ``atom_indices`` stores ``(source_stream_token, source_token_position)``
    for every retained atom.  It is the same provenance concept used by the
    vendored ITDA code, while allowing this pipeline to stream one prompt at a
    time instead of materialising a model-wide activation tensor.
    """

    def __init__(self, config: ITDAConfig) -> None:
        super().__init__()
        if config.d_model < 1 or config.max_atoms < 1 or config.k < 1:
            raise ValueError("ITDA dimensions and sparsity must be positive.")
        if config.k > config.max_atoms:
            raise ValueError("ITDA k cannot exceed max_atoms.")
        if not 0.0 <= float(config.loss_threshold) <= 4.0:
            raise ValueError("ITDA loss_threshold must be finite and in [0, 4].")
        self.config = config
        self.register_buffer("atoms", torch.empty((0, config.d_model)))
        self.register_buffer("atom_indices", torch.empty((0, 2), dtype=torch.long))

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.shape[0])

    @torch.no_grad()
    def _normalise(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        _require_finite(name, values)
        norms = values.norm(dim=-1, keepdim=True)
        if (norms <= 1e-8).any():
            raise ValueError(f"{name} contains a zero-norm vector.")
        result = values / norms
        _require_finite(f"normalised {name}", result)
        return result

    @torch.no_grad()
    def _append_atoms(self, values: torch.Tensor, source_indices: torch.Tensor) -> int:
        if values.ndim != 2 or values.shape[1] != self.config.d_model:
            raise ValueError("ITDA atoms must be shaped (atom, d_model).")
        if source_indices.shape != (values.shape[0], 2):
            raise ValueError("ITDA atom provenance must be shaped (atom, 2).")
        _require_finite("candidate ITDA atoms", values)
        remaining = self.config.max_atoms - self.n_atoms
        if remaining <= 0 or values.numel() == 0:
            return 0
        nonzero = values.norm(dim=-1) > 1e-8
        values = values[nonzero][:remaining]
        source_indices = source_indices[nonzero][:remaining]
        if values.numel() == 0:
            return 0
        self.atoms = torch.cat((self.atoms, self._normalise(values, name="ITDA atoms")), dim=0)
        self.atom_indices = torch.cat((self.atom_indices, source_indices.to(self.atom_indices.device)), dim=0)
        return int(values.shape[0])

    @torch.no_grad()
    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Return vendored-ITDA matching-pursuit coefficients."""

        if activations.ndim != 2 or activations.shape[1] != self.config.d_model:
            raise ValueError("ITDA activations must be shaped (token, d_model).")
        if self.n_atoms < 1:
            raise ValueError("ITDA cannot encode before at least one atom is added.")
        _require_finite("ITDA activations", activations)
        _require_finite("ITDA atoms", self.atoms)
        residual = activations.to(device=self.atoms.device, dtype=self.atoms.dtype).clone()
        coefficients = torch.zeros(
            (residual.shape[0], self.n_atoms), dtype=self.atoms.dtype, device=self.atoms.device
        )
        rows = torch.arange(residual.shape[0], device=self.atoms.device)
        for _ in range(min(self.config.k, self.n_atoms)):
            correlations = residual @ self.atoms.T
            best_atoms = correlations.abs().argmax(dim=1)
            values = correlations[rows, best_atoms]
            coefficients[rows, best_atoms] += values
            residual -= values.unsqueeze(1) * self.atoms[best_atoms]
        _require_finite("ITDA matching-pursuit coefficients", coefficients)
        _require_finite("ITDA matching-pursuit residual", residual)
        return coefficients

    @torch.no_grad()
    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != self.n_atoms:
            raise ValueError("ITDA coefficients must be shaped (token, n_atoms).")
        _require_finite("ITDA coefficients", coefficients)
        reconstruction = coefficients.to(self.atoms.dtype) @ self.atoms
        _require_finite("ITDA reconstruction", reconstruction)
        return reconstruction

    @torch.no_grad()
    def update(
        self,
        activations: torch.Tensor,
        *,
        source_indices: torch.Tensor | None = None,
    ) -> dict[str, float | int]:
        """Add source-order atoms whose normalised reconstruction error is high."""

        if activations.ndim != 2 or activations.shape[1] != self.config.d_model:
            raise ValueError("ITDA activations must be shaped (token, d_model).")
        _require_finite("ITDA fitting activations", activations)
        values = activations.detach().to(device=self.atoms.device, dtype=self.atoms.dtype)
        if source_indices is None:
            source_indices = torch.stack(
                (torch.arange(values.shape[0], device=values.device), torch.full((values.shape[0],), -1, device=values.device)),
                dim=1,
            )
        if source_indices.shape != (values.shape[0], 2):
            raise ValueError("ITDA source_indices must be shaped (token, 2).")
        source_indices = source_indices.to(device=self.atom_indices.device, dtype=torch.long)
        keep = values.norm(dim=-1) > 1e-8
        values, source_indices = values[keep], source_indices[keep]
        if values.numel() == 0:
            raise ValueError("ITDA fitting batch contains only zero-norm activations.")

        if self.n_atoms < self.config.d_model:
            # This matches the vendored initial basis construction: favour
            # repeated observed states, then retain source-order provenance.
            unique_values, inverse, counts = torch.unique(
                values, dim=0, return_inverse=True, return_counts=True
            )
            order = torch.argsort(counts, descending=True)
            missing = min(self.config.d_model - self.n_atoms, unique_values.shape[0])
            chosen = order[:missing]
            first_positions = torch.stack(
                [torch.nonzero(inverse == item, as_tuple=False)[0, 0] for item in chosen]
            )
            self._append_atoms(unique_values[chosen], source_indices[first_positions])

        coefficients = self.encode(values)
        reconstruction = self.decode(coefficients)
        normalised_values = self._normalise(values, name="ITDA fitting activations")
        normalised_reconstruction = self._normalise(reconstruction, name="ITDA reconstructions")
        errors = (normalised_values - normalised_reconstruction).square().mean(dim=1)
        _require_finite("ITDA reconstruction errors", errors)
        if (errors < 0).any() or (errors > 4.0 + 1e-5).any():
            raise ValueError("ITDA normalised reconstruction errors are outside [0, 4].")
        added = self._append_atoms(values[errors > self.config.loss_threshold], source_indices[errors > self.config.loss_threshold])
        return {
            "mean_normalised_mse": float(errors.mean().item()),
            "atoms_added": added,
            "n_atoms": self.n_atoms,
        }

    def save_pretrained(self, path: str | Path) -> None:
        torch.save(
            {
                "format": "mi_lens.itda.v1",
                "config": asdict(self.config),
                "atoms": self.atoms.detach().cpu(),
                "atom_indices": self.atom_indices.detach().cpu(),
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ITDA":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format") != "mi_lens.itda.v1":
            raise ValueError(f"{path} is not a mi_lens ITDA dictionary.")
        model = cls(ITDAConfig(**payload["config"]))
        model.atoms = payload["atoms"].to(map_location)
        model.atom_indices = payload["atom_indices"].to(map_location, dtype=torch.long)
        if model.n_atoms < 1 or model.n_atoms > model.config.max_atoms:
            raise ValueError("Serialized ITDA dictionary has an invalid atom count.")
        if model.atoms.shape != (model.n_atoms, model.config.d_model):
            raise ValueError("Serialized ITDA atoms have an invalid shape.")
        if model.atom_indices.shape != (model.n_atoms, 2):
            raise ValueError("Serialized ITDA provenance has an invalid shape.")
        _require_finite("serialized ITDA atoms", model.atoms)
        norms = model.atoms.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=2e-3, atol=2e-3):
            raise ValueError("Serialized ITDA atoms are not unit normalised.")
        return model
