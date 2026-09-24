"""The challenge metric: macro-averaged F0.5 over Source 1 entities."""


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

