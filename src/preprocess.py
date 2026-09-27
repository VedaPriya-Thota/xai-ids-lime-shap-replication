"""Feature fitting: TF-IDF bigrams + chi-squared selection, fit on training data only.

Used by the Phase 2.5 reconciliation diagnostics (train-only fit, never on validation/test) and
intended for reuse, unchanged, by the primary Phase 2/3 pipeline once it exists.
"""
from __future__ import annotations

import warnings
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def fit_features(
    train_texts: list[str], train_labels, other_texts: dict[str, list[str]], feat_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit a TF-IDF vectorizer and a chi-squared selector on `train_texts`/`train_labels` only,
    then transform every partition in `other_texts` (and "train" itself) with the fitted objects.

    Returns {"vectorizer", "selector", "matrices": {"train": ..., **other_texts-with-same-keys},
    "vocab_size": int, "k_actual": int}. If fewer bigrams exist than the configured k_best, all of
    them are kept (k_actual < k_best) and a UserWarning is raised so callers can flag it.
    """
    ngram_range = tuple(feat_cfg.get("ngram_range", [2, 2]))
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        lowercase=bool(feat_cfg.get("tfidf_lowercase", False)),
        token_pattern=feat_cfg.get("tfidf_token_pattern", r"\S+"),
        use_idf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_texts)
    vocab_size = int(x_train.shape[1])

    k_requested = int(feat_cfg.get("k_best", 150))
    k_actual = min(k_requested, vocab_size)
    if k_actual < k_requested:
        warnings.warn(
            f"fewer than the requested {k_requested} features are available ({vocab_size} observed "
            f"bigrams); using k={k_actual}",
            stacklevel=2,
        )

    matrices: dict[str, Any] = {}
    if k_actual > 0:
        selector = SelectKBest(chi2, k=k_actual)
        matrices["train"] = selector.fit_transform(x_train, train_labels)
    else:
        selector = None
        matrices["train"] = x_train

    for name, texts in other_texts.items():
        x = vectorizer.transform(texts)
        matrices[name] = selector.transform(x) if selector is not None else x

    return {
        "vectorizer": vectorizer, "selector": selector, "matrices": matrices,
        "vocab_size": vocab_size, "k_actual": k_actual,
    }
