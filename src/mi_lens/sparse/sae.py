"""Small Top-K SAE implementation used by the RouterInterp workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(slots=True)
class TopKSAEConfig:
    d_model: int
    n_features: int
    k: int


class TopKSAE(nn.Module):
    """Sparse autoencoder with exactly ``k`` non-zero latent activations."""

    def __init__(self, config: TopKSAEConfig) -> None:
        super().__init__()
        if not 1 <= config.k <= config.n_features:
            raise ValueError("`k` must be in [1, n_features].")
        self.config = config
        self.encoder = nn.Linear(config.d_model, config.n_features)
        self.decoder = nn.Linear(config.n_features, config.d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(config.d_model))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.shape[-1] != self.config.d_model:
            raise ValueError(
                f"Expected d_model={self.config.d_model}, got {activations.shape[-1]}."
            )
        dense = torch.relu(self.encoder(activations - self.decoder_bias))
        values, indices = dense.topk(k=self.config.k, dim=-1)
        sparse = torch.zeros_like(dense)
        return sparse.scatter(-1, indices, values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.decoder_bias

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(activations)
        return self.decode(features), features

    def save_pretrained(self, path: str) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def from_pretrained(cls, path: str, *, map_location: str | torch.device = "cpu") -> "TopKSAE":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(TopKSAEConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


def fit_topk_sae(
    activations: torch.Tensor,
    config: TopKSAEConfig,
    *,
    steps: int = 10_000,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> tuple[TopKSAE, list[float]]:
    """Train a Top-K SAE on a token-by-residual activation matrix."""

    if activations.ndim != 2:
        raise ValueError("Expected activations with shape (token, d_model).")
    if activations.shape[0] < 1:
        raise ValueError("Need at least one activation to train an SAE.")
    if activations.shape[1] != config.d_model:
        raise ValueError("Activation width does not match SAE d_model.")
    if steps < 1 or batch_size < 1:
        raise ValueError("`steps` and `batch_size` must be positive.")

    model = TopKSAE(config).to(activations.device, dtype=activations.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=activations.device).manual_seed(seed)
    history: list[float] = []
    for _ in range(steps):
        indices = torch.randint(
            activations.shape[0],
            (min(batch_size, activations.shape[0]),),
            device=activations.device,
            generator=generator,
        )
        batch = activations[indices]
        reconstruction, _ = model(batch)
        loss = torch.nn.functional.mse_loss(reconstruction, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        history.append(float(loss.detach().item()))
    return model.eval(), history
