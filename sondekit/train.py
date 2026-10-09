"""Probe fitting, thresholds and numpy metrics."""

from __future__ import annotations

import collections
import copy
import logging
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import torch

from sondekit import config
from sondekit import data
from sondekit import probe
from sondekit import probes

SIGMA_FLOOR = 1e-6
_LBFGS_ITERS = 500

_log = logging.getLogger(__name__)


def _rows(X: torch.Tensor, offsets: torch.Tensor | None, i: int):
    """[H] for pooled, [T, H] for tokens."""
    return X[i] if offsets is None else X[offsets[i] : offsets[i + 1]]


def pad(
    X: torch.Tensor, offsets: torch.Tensor | None, idx: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gathers samples idx into a float32 batch.

    Args:
        X: [n, H] pooled features, or flat [sum T, H] tokens.
        offsets: None for pooled, else [n + 1] row offsets into X.
        idx: [B] sample indices.

    Returns:
        (x, mask): x [B, H] and None for pooled; x [B, T, H] right-padded
        with zeros and mask [B, T] bool for tokens.
    """
    if offsets is None:
        return X[idx].float(), None
    rows = [_rows(X, offsets, i).float() for i in idx]
    x = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    n = torch.tensor([len(r) for r in rows])
    return x, torch.arange(x.shape[1])[None, :] < n[:, None]


def standardise(
    X: torch.Tensor, offsets: torch.Tensor | None, train_idx: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-split feature statistics.

    Args:
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1] token offsets.
        train_idx: train sample indices.

    Returns:
        (mu [H], sigma [H]) float32; sigma floored at SIGMA_FLOOR. Token data
        uses every window token of the train samples.
    """
    if offsets is None:
        rows = X[train_idx].float()
    else:
        rows = torch.cat([_rows(X, offsets, i) for i in train_idx]).float()
    return rows.mean(0), rows.std(0).clamp_min(SIGMA_FLOOR)


def control_passed(
    control_aurocs: Sequence[float], n_pos: int, n_neg: int
) -> bool:
    """Whether shuffled-label AUROCs are consistent with chance.

    Two checks. Leakage shows as shuffles off chance in the same direction,
    so the mean must sit within max(2 standard errors, 0.05) of 0.5. And
    each shuffle must sit within max(4 sd0, 0.05) of 0.5, where sd0 is the
    AUROC's standard deviation under label independence (Mann-Whitney). A
    random-label fit beyond that separates the classes along a dominant
    direction of variance with a random sign, so a high probe score does
    not show that the probe learned anything specific to the label.

    Args:
        control_aurocs: Validation AUROC of each shuffled-label fit.
        n_pos: Positive validation rows.
        n_neg: Negative validation rows.

    Returns:
        True when both checks pass.
    """
    a = np.asarray(control_aurocs, dtype=np.float64)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    sd0 = np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
    return bool(
        abs(a.mean() - 0.5) <= max(2 * se, 0.05)
        and np.abs(a - 0.5).max() <= max(4 * sd0, 0.05)
    )


def diff_means(
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    labels: np.ndarray,
    train_idx: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Raw-space difference of class means, scaled to unit-std logits.

    The raw w = mu_pos - mu_neg gives logits of order |w|^2 (hundreds on a
    residual stream), which saturates the float32 sigmoid and leaves the
    threshold nothing to choose between. Dividing by the train logits' std
    keeps the direction and the ranking.

    Args:
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1]; token samples are mean-pooled first.
        labels: [n] in {0, 1}.
        train_idx: train sample indices.

    Returns:
        (w [H] float64, b) with b = -w . (mu_pos + mu_neg) / 2.
    """
    if offsets is None:
        feats = X[train_idx].double().numpy()
    else:
        feats = torch.stack(
            [_rows(X, offsets, i).double().mean(0) for i in train_idx]
        ).numpy()
    y = labels[train_idx]
    pos, neg = feats[y == 1].mean(0), feats[y == 0].mean(0)
    w = pos - neg
    w = w / max(float(np.std(feats @ w)), SIGMA_FLOOR)
    return w, float(-w @ (pos + neg) / 2)


def _lbfgs(module, X, y, tr, mu, sigma, loss_fn, l2: float) -> None:
    """Full-batch L-BFGS to convergence for a linear probe on pooled rows.

    The problem is convex logistic regression, so a fixed AdamW step budget
    stops far from the optimum on small or easy data. l2 * |w|^2 keeps the
    weights finite when the classes are separable.
    """
    x = (X[tr].to(mu.device).float() - mu) / sigma
    target = y[tr]
    opt = torch.optim.LBFGS(
        module.parameters(),
        max_iter=_LBFGS_ITERS,
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(module(x), target)
        loss = loss + l2 * module.linear.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)


def fit(
    cfg: config.ProbeConfig,
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    labels: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    seed: int,
) -> tuple[
    probes.LinearProbe | probes.AttentionProbe, torch.Tensor, torch.Tensor
]:
    """Trains one probe on one block.

    AdamW on CUDA when available. The seed is set before the module is
    built; a local generator shuffles batches. pos_weight = n_neg / n_pos;
    weight decay skips biases. Early stopping on val loss with
    cfg.patience (0 = off); the best epoch's weights are returned.

    Args:
        cfg: probe section.
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1].
        labels: [n] in {0, 1}.
        train_idx: train sample indices.
        val_idx: val sample indices.
        seed: run seed.

    Returns:
        (module on CPU, mu [H], sigma [H]); diff_means returns mu = 0 and
        sigma = 1 so export leaves its raw-space weights untouched.
    """
    hidden = X.shape[1]
    torch.manual_seed(seed)
    module = probes.build(cfg, hidden)
    if cfg.init == "diff_means":
        w, b = diff_means(X, offsets, labels, train_idx)
        with torch.no_grad():
            module.linear.weight[0] = torch.from_numpy(w)
            module.linear.bias[0] = b
        return module, torch.zeros(hidden), torch.ones(hidden)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)
    mu, sigma = standardise(X, offsets, train_idx)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    n_pos = float(labels[tr].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(tr) - n_pos) / n_pos, device=device)
    )
    bias = module.linear.bias
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in module.parameters() if p is not bias]},
            {"params": [bias], "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    gen = torch.Generator().manual_seed(seed)
    mu_d, sigma_d = mu.to(device), sigma.to(device)
    if cfg.kind == "linear" and offsets is None:
        _lbfgs(module, X, y, tr, mu_d, sigma_d, loss_fn, cfg.weight_decay)
        return module.cpu(), mu, sigma
    val_chunks = [
        va[i : i + cfg.batch_size] for i in range(0, len(va), cfg.batch_size)
    ]

    def loss_on(idx: np.ndarray) -> torch.Tensor:
        x, mask = pad(X, offsets, idx)
        x = (x.to(device) - mu_d) / sigma_d
        mask = None if mask is None else mask.to(device)
        return loss_fn(module(x, mask), y[idx])

    best, best_loss, bad = copy.deepcopy(module.state_dict()), np.inf, 0
    for _ in range(cfg.epochs):
        for b in torch.randperm(len(tr), generator=gen).split(cfg.batch_size):
            loss = loss_on(tr[b.numpy()])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_loss = sum(
                float(loss_on(c)) * len(c) for c in val_chunks
            ) / len(va)
        if val_loss < best_loss:
            best, best_loss = copy.deepcopy(module.state_dict()), val_loss
            bad = 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break
    module.load_state_dict(best)
    return module.cpu(), mu, sigma


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC; tied scores share their mean rank.

    Raises:
        ValueError: y lacks a class.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("auroc needs both classes")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _counts(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    """(tp, fp) at each threshold for the rule score >= threshold."""
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return tp, fp


def choose_threshold(
    y: np.ndarray, scores: np.ndarray, max_fpr: float | None
) -> tuple[float, bool]:
    """Operating threshold on probe scores in [0, 1].

    With max_fpr: the lowest score threshold whose FPR <= max_fpr. When
    none qualifies (negatives tie at the top score), falls back to best F1
    and reports failure. Without max_fpr: the best-F1 threshold.

    Returns:
        (threshold, threshold_failed).
    """
    cand = np.unique(scores)
    tp, fp = _counts(y, scores, cand)
    if max_fpr is not None:
        ok = fp / max(int((y == 0).sum()), 1) <= max_fpr
        if ok.any():
            return float(cand[ok.argmax()]), False
    f1 = 2 * tp / (tp + fp + y.sum())
    return float(cand[f1.argmax()]), max_fpr is not None


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Recall at this set's own lowest threshold with FPR <= max_fpr.

    Args:
        y: [n] labels with both classes.
        scores: [n] scores.
        max_fpr: the FPR budget.

    Returns:
        The recall; 0.0 when no threshold meets the budget.
    """
    thr, failed = choose_threshold(y, scores, max_fpr)
    return 0.0 if failed else float((scores[y == 1] >= thr).mean())


def group_auroc(
    y: np.ndarray, scores: np.ndarray, groups: Sequence
) -> tuple[float | None, int]:
    """Mean AUROC within each group that holds both classes.

    GRPO's group-relative advantage only uses the ranking among one
    prompt's completions, so this is the headline for reward probes.

    Args:
        y: [n] labels in {0, 1}.
        scores: [n] scores.
        groups: [n] group keys.

    Returns:
        (mean within-group AUROC, n_groups); (None, 0) when no group holds
        both classes.
    """
    members = collections.defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    vals = [
        auroc(y[i], scores[i])
        for i in members.values()
        if 0 < y[i].sum() < len(i)
    ]
    return (float(np.mean(vals)) if vals else None), len(vals)


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    n: int = 1000,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of fn(y, scores).

    Args:
        y: [m] labels with both classes.
        scores: [m] scores.
        fn: the statistic, e.g. auroc.
        seed: numpy seed for the resamples.
        n: resamples; those that lose a class are skipped.

    Returns:
        (low, high).
    """
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            stats.append(fn(y[i], scores[i]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_fpr: float | None,
    groups: Sequence | None = None,
) -> dict:
    """Eval card for one labelled set. Never raises on a single class.

    A benign-only set reports fpr only and a positive-only set recall
    only; metrics that need both classes are None.

    Args:
        y: [n] labels in {0, 1}.
        scores: [n] scores.
        threshold: the operating threshold.
        max_fpr: FPR budget for recall_at_fpr and small_n, or None.
        groups: [n] group keys, or None to skip group_auroc.

    Returns:
        n_pos, n_neg, threshold, auroc, f1, precision, recall and fpr at
        threshold, small_n (n_neg * max_fpr < 10), recall_at_fpr when
        max_fpr is set, and group_auroc with n_groups when groups is set.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    both = n_pos > 0 and n_neg > 0
    (tp,), (fp,) = _counts(y, scores, np.array([threshold]))
    out = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold),
        "auroc": auroc(y, scores) if both else None,
        "f1": float(2 * tp / (tp + fp + n_pos)) if both else None,
        "precision": float(tp / max(tp + fp, 1)) if both else None,
        "recall": float(tp / n_pos) if n_pos else None,
        "fpr": float(fp / n_neg) if n_neg else None,
        "small_n": max_fpr is not None and n_neg * max_fpr < 10,
    }
    if out["small_n"]:
        _log.warning(
            "small_n: %d negatives x max_fpr %s < 10; the FPR threshold "
            "rests on too few negatives",
            n_neg,
            max_fpr,
        )
    if max_fpr is not None:
        out["recall_at_fpr"] = (
            recall_at_fpr(y, scores, max_fpr) if both else None
        )
    if groups is not None:
        out["group_auroc"], out["n_groups"] = group_auroc(y, scores, groups)
    return out


def probe_scores(
    p: probe.Probe,
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    idx: Iterable[int],
) -> np.ndarray:
    """[len(idx)] float32 scores from the numpy artifact's pooled_score."""
    return np.array(
        [p.pooled_score(_rows(X, offsets, i).float().numpy()) for i in idx],
        dtype=np.float32,
    )


def bow_features(
    texts: list[str], train_idx: Iterable[int], vocab_size: int = 5000
) -> np.ndarray:
    """Lowercase word counts over the train split's most frequent words.

    Args:
        texts: [n] window texts.
        train_idx: train sample indices; only they build the vocabulary.
        vocab_size: words kept.

    Returns:
        [n, V] float32 counts, V <= vocab_size.
    """
    words = [re.findall(r"\w+", t.lower()) for t in texts]
    top = collections.Counter(w for i in train_idx for w in words[i])
    vocab = {w: j for j, (w, _) in enumerate(top.most_common(vocab_size))}
    # NOTE: dense [n, V]; use scipy.sparse if n * V outgrows RAM.
    out = np.zeros((len(texts), len(vocab)), np.float32)
    for i, row in enumerate(words):
        for w in row:
            if w in vocab:
                out[i, vocab[w]] += 1
    return out


def window_text(sample: data.Sample, window: str) -> str:
    """The text a window covers, for the bag-of-words baseline.

    Args:
        sample: a loaded row.
        window: one of probe.WINDOWS.

    Returns:
        The prompt for "prompt", the response for "response", both for
        "all", and the last message for "last_turn".
    """
    turns = [m["content"] for m in sample.messages or []] or [sample.text]
    prompt = "\n".join(t or "" for t in turns)
    if window == "prompt":
        return prompt
    if window == "response":
        return sample.response or ""
    if window == "all":
        return f"{prompt}\n{sample.response or ''}"
    return turns[-1] or ""
