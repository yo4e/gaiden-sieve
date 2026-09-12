"""Phase 1 training loop: split, vectorize, fit, and evaluate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from gaiden_sieve.data import LabeledItem, build_text
from gaiden_sieve.evaluate import Metrics, calculate_metrics
from gaiden_sieve.profile import ProfileConfig


RANDOM_SEED = 42
TEST_SIZE = 0.20


class TrainingError(ValueError):
    """Raised when labeled data cannot support the Phase 1 training contract."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """In-memory model plus the holdout score used during Phase 1."""

    model: Pipeline
    metrics: Metrics
    train_size: int
    test_size: int
    random_seed: int


def build_pipeline(*, random_seed: int = RANDOM_SEED) -> Pipeline:
    """Create one preprocessing + classifier pipeline shared by train and inference."""

    # 学習時と推論時で同じ前処理を必ず通すため、TF-IDF と分類器を1本の Pipeline にまとめる。
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2)),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_seed),
            ),
        ]
    )


def _validate_training_data(items: Sequence[LabeledItem], *, test_size: float) -> None:
    label_counts = Counter(item.label for item in items)
    if set(label_counts) != {"relevant", "not_relevant"}:
        raise TrainingError("training data must contain relevant and not_relevant labels")
    if min(label_counts.values()) < 2:
        raise TrainingError("each label needs at least two items for a stratified split")

    test_count = ceil(len(items) * test_size)
    train_count = len(items) - test_count
    if test_count < len(label_counts) or train_count < len(label_counts):
        raise TrainingError("train/test split is too small to contain both labels")


def train_and_evaluate(
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """Train on one stratified split and score only the held-out test rows."""

    if not 0.0 < test_size < 1.0:
        raise TrainingError("test_size must be between 0 and 1")
    _validate_training_data(items, test_size=test_size)

    texts = [build_text(item, profile.text_fields) for item in items]
    labels = [item.label for item in items]

    # test データはモデルへ見せず、学習後の成績確認だけに使う。
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    model = build_pipeline(random_seed=random_seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = calculate_metrics(
        y_test,
        y_pred,
        positive_label=profile.positive_label,
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(x_train),
        test_size=len(x_test),
        random_seed=random_seed,
    )
