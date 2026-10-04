from __future__ import annotations

import numpy as np
import pytest
import torch

from sonde import config
from sonde import probes
from sonde import train

CASES = [
    ("linear", "mean", None),
    ("linear", "last", None),
    ("linear", "max", None),
    ("linear", "rolling_mean", 3),
    ("linear", "rolling_mean", 9),
    ("attention", "attention", None),
]
META = dict(
    name="p",
    block=2,
    window="prompt",
    threshold=0.5,
    model="m",
    engine="hf==0",
    model_fingerprint=None,
    prompt_format="raw",
)


@pytest.mark.parametrize("kind,pooling,rw", CASES)
def test_torch_matches_numpy_probe(kind, pooling, rw):
    torch.manual_seed(0)
    cfg = config.ProbeConfig(kind=kind, pooling=pooling, rolling_window=rw)
    module = probes.build(cfg, 8)
    lengths = [1, 4, 7]
    X = torch.randn(sum(lengths), 8) * 3 + 1
    offsets = torch.tensor([0, 1, 5, 12])
    mu, sigma = X.mean(0), X.std(0)
    x, mask = train.pad(X, offsets, [0, 1, 2])
    with torch.no_grad():
        want = torch.sigmoid(module((x - mu) / sigma, mask)).numpy()
    p = probes.export(module, mu, sigma, **META)
    got = [
        p.pooled_score(X[offsets[i] : offsets[i + 1]].numpy()) for i in range(3)
    ]
    assert p.kind == kind and p.pooling == pooling
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_pooled_input_matches_numpy_probe():
    torch.manual_seed(0)
    module = probes.LinearProbe(8, "mean")
    X = torch.randn(5, 8)
    mu, sigma = X.mean(0), X.std(0)
    x, mask = train.pad(X, None, [0, 3])
    assert mask is None and x.shape == (2, 8)
    with torch.no_grad():
        want = torch.sigmoid(module((x - mu) / sigma)).numpy()
    p = probes.export(module, mu, sigma, **META)
    got = [p.pooled_score(X[i].numpy()) for i in (0, 3)]
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_pad_shapes():
    X = torch.arange(12.0).reshape(6, 2)
    x, mask = train.pad(X, torch.tensor([0, 2, 6]), [1, 0])
    assert x.shape == (2, 4, 2) and x.dtype == torch.float32
    assert mask is not None
    assert mask.tolist() == [[True] * 4, [True, True, False, False]]
    assert x[1, :2].tolist() == [[0.0, 1.0], [2.0, 3.0]]
