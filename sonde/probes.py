"""Torch probe modules that compute exactly what `probe.Probe` scores."""

from __future__ import annotations

import torch

from sonde import config
from sonde import probe


def _pool(
    z: torch.Tensor,
    mask: torch.Tensor,
    pooling: str,
    rolling_window: int | None,
) -> torch.Tensor:
    """Pools per-token logits z [B, T] over the valid tokens of mask [B, T]."""
    n = mask.sum(1)
    if pooling == "last":
        return z.gather(1, (n - 1)[:, None])[:, 0]
    if pooling == "max":
        return z.masked_fill(~mask, -torch.inf).max(1).values
    zm = z * mask
    mean = zm.sum(1) / n
    w = rolling_window
    if pooling == "mean" or w is None or z.shape[1] < w:
        return mean
    c = torch.nn.functional.pad(zm.cumsum(1), (1, 0))
    win = (c[:, w:] - c[:, :-w]) / w
    start = torch.arange(win.shape[1], device=z.device)
    win = win.masked_fill(start + w > n[:, None], -torch.inf)
    return torch.where(n < w, mean, win.max(1).values)


class LinearProbe(torch.nn.Module):
    """Linear logit per token, pooled: mean | last | max | rolling_mean."""

    def __init__(
        self, hidden: int, pooling: str, rolling_window: int | None = None
    ):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 1)
        self.pooling = pooling
        self.rolling_window = rolling_window

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pooled logits.

        Args:
            x: [B, H] pre-pooled features (mask None), or [B, T, H] tokens.
            mask: [B, T] bool, True on valid tokens, right-padded.

        Returns:
            [B] logits.
        """
        z = self.linear(x)[..., 0]
        if mask is None:
            return z
        return _pool(z, mask, self.pooling, self.rolling_window)


class AttentionProbe(torch.nn.Module):
    """softmax(x @ q) weighted sum of per-token logits x @ w + b."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 1)
        self.q = torch.nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-pooled logits.

        Args:
            x: [B, T, H] tokens.
            mask: [B, T] bool, True on valid tokens.

        Returns:
            [B] logits.
        """
        a = self.q(x)[..., 0].masked_fill(~mask, -torch.inf)
        return (torch.softmax(a, 1) * self.linear(x)[..., 0]).sum(1)


def build(cfg: config.ProbeConfig, hidden: int) -> LinearProbe | AttentionProbe:
    """Builds the untrained module that `cfg` describes."""
    if cfg.kind == "attention":
        return AttentionProbe(hidden)
    return LinearProbe(hidden, cfg.pooling, cfg.rolling_window)


def export(
    module: LinearProbe | AttentionProbe,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    **meta,
) -> probe.Probe:
    """Folds standardisation into raw-space weights and builds the artifact.

    The module saw (x - mu) / sigma, so w' = w / sigma and
    b' = b - sum(w * mu / sigma). q folds to q / sigma; its constant term is
    dropped because softmax is shift-invariant.

    Args:
        module: trained LinearProbe or AttentionProbe.
        mu: [H] train mean.
        sigma: [H] floored train std.
        **meta: the remaining Probe fields (name, block, window, threshold,
            model, engine, model_fingerprint, prompt_format, metrics, ...).

    Returns:
        The numpy Probe.
    """
    w = module.linear.weight.detach()[0].double()
    mu, sigma = mu.double(), sigma.double()
    bias = module.linear.bias.item() - float(w @ (mu / sigma))
    attention = isinstance(module, AttentionProbe)
    q = None
    if attention:
        q = (module.q.weight.detach()[0].double() / sigma).numpy()
    return probe.Probe(
        kind="attention" if attention else "linear",
        w=(w / sigma).numpy(),
        bias=bias,
        pooling="attention" if attention else module.pooling,
        q=q,
        rolling_window=None if attention else module.rolling_window,
        **meta,
    )
