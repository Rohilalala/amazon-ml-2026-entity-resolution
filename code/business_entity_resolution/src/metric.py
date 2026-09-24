"""The challenge metric: macro-averaged F0.5 over Source 1 entities.

`f05_entity` / `f05_macro` follow the problem statement literally (sets of ids);
`f05_counts` is the vectorised form used on millions of entities, and the tests
check that both agree.
"""
import numpy as np


def f05_entity(pred, truth):
    """F0.5 for one Source 1 entity. Singletons score 1.0 only for an empty prediction."""
    if not truth:
        return 0.0 if pred else 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(pred, truth, ids):
    """Mean F0.5 over `ids`; `pred`/`truth` map a Source 1 id to a set of matched ids."""
    empty = frozenset()
    return sum(f05_entity(pred.get(i, empty), truth.get(i, empty)) for i in ids) / len(ids)


def f05_counts(n_pred, n_true, tp):
    """Per-entity F0.5 from arrays of predicted, true and correct match counts."""
    n_pred, n_true, tp = (np.asarray(x, dtype=np.float64) for x in (n_pred, n_true, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)
