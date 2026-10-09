from __future__ import annotations

import numpy as np
import pytest
import torch

from sondekit import config
from sondekit import data
from sondekit import probes
from sondekit import train

META = dict(
    name="p",
    block=0,
    window="prompt",
    threshold=0.5,
    model="m",
    engine="hf==0",
    model_fingerprint=None,
    prompt_format="raw",
)


def _data():
    g = torch.Generator().manual_seed(0)
    y = np.arange(80) % 2
    X = torch.randn(80, 4, generator=g) * 5 + 10
    X[:, 0] += torch.as_tensor(y, dtype=torch.float32) * 15
    return X, y


def test_auroc_rank_based_with_ties():
    y = np.array([0, 0, 1, 1])
    assert train.auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert train.auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert train.auroc(y, np.array([0.1, 0.2, 0.3, 0.3])) == 1.0


def test_threshold_at_max_fpr():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    assert train.choose_threshold(y, s, 0.25) == (0.4, False)
    assert train.choose_threshold(y, s, 0.0) == (0.8, False)


def test_threshold_failed_falls_back_to_best_f1():
    y = np.array([0, 1, 1])
    s = np.array([0.9, 0.9, 0.2])
    thr, failed = train.choose_threshold(y, s, 0.0)
    assert failed and thr == 0.2


def test_best_f1_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.5, 0.9])
    assert train.choose_threshold(y, s, None) == (0.5, False)


def test_metrics_counts():
    y = np.array([0, 0, 0, 1, 1])
    m = train.metrics(y, np.array([0.1, 0.2, 0.9, 0.8, 0.95]), 0.5, 0.0)
    assert (m["n_pos"], m["n_neg"]) == (2, 3)
    assert m["recall"] == 1.0 and m["fpr"] == 1 / 3
    assert m["precision"] == 2 / 3 and m["recall_at_fpr"] == 0.5
    assert train.metrics(np.zeros(3), np.ones(3), 0.5, None)["auroc"] is None


def test_recall_at_fpr():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    assert train.recall_at_fpr(y, s, 0.25) == 1.0
    assert train.recall_at_fpr(y, s, 0.0) == 0.5


def test_single_class_reports_one_side_without_raising():
    s = np.array([0.1, 0.2, 0.6, 0.9])
    benign = train.metrics(np.zeros(4), s, 0.5, 0.01)
    assert benign["fpr"] == 0.5 and benign["recall"] is None
    assert benign["auroc"] is None and benign["recall_at_fpr"] is None
    positive = train.metrics(np.ones(4), s, 0.5, 0.01)
    assert positive["recall"] == 0.5 and positive["fpr"] is None


def test_small_n_flag(caplog):
    y = np.r_[np.zeros(2000), np.ones(10)]
    assert not train.metrics(y, y, 0.5, 0.01)["small_n"]
    assert not train.metrics(y, y, 0.5, None)["small_n"]
    assert train.metrics(y[1500:], y[1500:], 0.5, 0.01)["small_n"]
    assert "small_n: 500 negatives" in caplog.text


def test_group_auroc_on_hand_built_groups():
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.2, 0.5, 0.6, 0.3, 0.4])
    g = ["a", "a", "b", "b", "c", "c", "d", "d"]
    assert train.group_auroc(y, s, g) == (0.5, 2)
    assert train.group_auroc(y[4:], s[4:], g[4:])[1] == 0
    m = train.metrics(y, s, 0.5, None, groups=g)
    assert (m["group_auroc"], m["n_groups"]) == (0.5, 2)


def test_bootstrap_ci_contains_point_estimate():
    y = np.arange(200) % 2
    s = y + np.random.default_rng(0).normal(size=200)
    low, high = train.bootstrap_ci(y, s, train.auroc, seed=0)
    assert low < train.auroc(y, s) < high
    assert train.bootstrap_ci(y, s, train.auroc, seed=0) == (low, high)


def test_sigma_floor():
    X = torch.ones(6, 3)
    mu, sigma = train.standardise(X, None, [0, 1, 2])
    assert torch.all(sigma == train.SIGMA_FLOOR) and torch.all(mu == 1)


def test_standardise_tokens_uses_train_tokens_only():
    X = torch.tensor([[1.0], [3.0], [100.0]])
    mu, _ = train.standardise(X, torch.tensor([0, 2, 3]), [0])
    assert mu.item() == 2.0


def test_standardisation_fold_is_exact():
    X, y = _data()
    cfg = config.ProbeConfig(epochs=3, lr=1e-2, batch_size=16)
    module, mu, sigma = train.fit(cfg, X, None, y, range(60), range(60, 80), 0)
    p = probes.export(module, mu, sigma, **META)
    with torch.no_grad():
        want = torch.sigmoid(module((X - mu) / sigma)).numpy()
    got = train.probe_scores(p, X, None, range(80))
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_fit_is_seed_reproducible_and_learns():
    X, y = _data()
    cfg = config.ProbeConfig(epochs=20, lr=1e-2, batch_size=16, patience=0)
    a, _, _ = train.fit(cfg, X, None, y, range(60), range(60, 80), 7)
    b, _, _ = train.fit(cfg, X, None, y, range(60), range(60, 80), 7)
    assert torch.equal(a.linear.weight, b.linear.weight)
    with torch.no_grad():
        s = a((X[60:] - X[:60].mean(0)) / X[:60].std(0)).numpy()
    assert train.auroc(y[60:], s) > 0.9


def test_diff_means_closed_form():
    X = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 1, 0, 0])
    w, b = train.diff_means(X, None, y, [0, 1, 2, 3])
    np.testing.assert_allclose(w / np.linalg.norm(w), [2**-0.5, 2**-0.5])
    pos, neg = X[:2].double().mean(0).numpy(), X[2:].double().mean(0).numpy()
    assert b == pytest.approx(-w @ (pos + neg) / 2)
    cfg = config.ProbeConfig(init="diff_means", epochs=0)
    module, mu, sigma = train.fit(cfg, X, None, y, [0, 1, 2, 3], [0], 0)
    p = probes.export(module, mu, sigma, **META)
    np.testing.assert_allclose(p.w, w, rtol=1e-6)
    assert p.bias == pytest.approx(b)


def test_diff_means_logits_have_unit_train_std():
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(0, 30, (200, 64)).astype(np.float32))
    y = rng.integers(0, 2, 200)
    X[y == 1] += 5.0
    w, _ = train.diff_means(X, None, y, range(200))
    assert np.std(X.double().numpy() @ w) == pytest.approx(1.0)


def test_diff_means_tokens_mean_pools_each_sample():
    X = torch.tensor([[0.0], [2.0], [5.0]])
    w, _ = train.diff_means(
        X, torch.tensor([0, 2, 3]), np.array([0, 1]), [0, 1]
    )
    assert w[0] > 0


def test_bow_baseline_separates_keyword_only_data():
    np.testing.assert_array_equal(
        train.bow_features(["Bomb bomb!", "x"], [0]), [[2.0], [0.0]]
    )
    rng = np.random.default_rng(0)
    filler = ["the", "a", "cat", "dog", "ran", "sat", "on", "mat"]
    y = np.arange(80) % 2
    texts = [
        " ".join(
            [*rng.choice(filler, 6), "bomb"] if label else rng.choice(filler, 7)
        )
        for label in y
    ]
    X = torch.from_numpy(train.bow_features(texts, range(60)))
    assert X.shape == (80, 9)
    cfg = config.ProbeConfig(epochs=30, lr=1e-2, batch_size=16)
    module, mu, sigma = train.fit(cfg, X, None, y, range(60), range(60, 80), 0)
    with torch.no_grad():
        s = module((X[60:] - mu) / sigma).numpy()
    assert train.auroc(y[60:], s) == 1.0


def test_window_text():
    s = data.Sample(
        id="0",
        text=None,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
        response="ok",
        response_ids=None,
        label=1,
        group=None,
        raw={},
    )
    assert train.window_text(s, "prompt") == "hi\nyo"
    assert train.window_text(s, "response") == "ok"
    assert train.window_text(s, "all") == "hi\nyo\nok"
    assert train.window_text(s, "last_turn") == "yo"


def test_control_is_a_test_against_chance():
    assert train.control_passed([0.5, 0.52, 0.48, 0.51, 0.49], 78, 78)
    # Consistently off chance, in either direction, fails.
    assert not train.control_passed([0.84, 0.86, 0.85, 0.83, 0.87], 78, 78)
    assert not train.control_passed([0.15, 0.17, 0.16, 0.14, 0.18], 78, 78)


def test_control_fails_when_random_fits_separate_the_classes():
    # A diff-means run whose shuffles averaged 0.67: the label is the dominant
    # direction of variance, so a random-label fit finds it with either sign.
    a = [0.263, 0.365, 0.838, 0.947, 0.945]
    assert not train.control_passed(a, 78, 78)


def test_control_spread_bound_scales_with_validation_size():
    a = [0.8, 0.2, 0.75, 0.3, 0.81]
    assert train.control_passed(a, 8, 8)
    assert not train.control_passed(a, 78, 78)


def test_fit_recovers_one_perfect_feature_among_noise():
    rng = np.random.default_rng(0)
    y = (rng.random(1400) < 0.16).astype(np.int64)
    X = rng.integers(0, 2, (1400, 25)).astype(np.float32)
    X[:, 7] = y
    X = torch.from_numpy(X)
    module, mu, sigma = train.fit(
        config.ProbeConfig(), X, None, y, range(1000), range(1000, 1400), 0
    )
    p = probes.export(module, mu, sigma, **META)
    s = train.probe_scores(p, X, None, range(1000, 1400))
    assert train.auroc(y[1000:], s) > 0.99


def test_pooled_linear_fit_converges():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    X = torch.from_numpy(rng.normal(size=(300, 16)).astype(np.float32))
    X[:, 0] += torch.from_numpy(y.astype(np.float32)) * 3.0
    a, *_ = train.fit(
        config.ProbeConfig(), X, None, y, range(200), range(200, 300), 0
    )
    b, *_ = train.fit(
        config.ProbeConfig(epochs=2000, patience=0),
        X,
        None,
        y,
        range(200),
        range(200, 300),
        0,
    )
    # Default settings reach the same optimum as a 100x longer run.
    assert torch.allclose(a.linear.weight, b.linear.weight, atol=1e-3)
