"""Training plus candidate creation for the GAIDEN SIEVE model lifecycle."""

from __future__ import annotations

import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Sequence

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from gaiden_sieve.artifacts import (
    ArtifactError,
    load_candidate_model,
    resolve_code_commit,
    save_candidate_metadata,
    save_candidate_model,
    sha256_file,
    verify_candidate_integrity,
)
from gaiden_sieve.data import (
    LabeledItem,
    build_text,
    labeled_dataset_hash,
    load_labeled_jsonl,
)
from gaiden_sieve.evaluate import Metrics, calculate_metrics, evaluate_quality_gate
from gaiden_sieve.profile import ProfileConfig


RANDOM_SEED = 42
TEST_SIZE = 0.20


class TrainingError(ValueError):
    """Raised when labeled data cannot support the training contract."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """In-memory model plus the holdout score used during training."""

    model: Pipeline
    metrics: Metrics
    train_size: int
    test_size: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class _Split:
    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]


def build_pipeline(*, random_seed: int = RANDOM_SEED) -> Pipeline:
    """Create one preprocessing + classifier pipeline shared by train and inference."""

    # 学習時と推論時で同じ前処理を必ず通すため、TF-IDF と分類器を1本の Pipeline にまとめる。
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_seed),
            ),
        ]
    )


def _label_counts(items: Sequence[LabeledItem], *, dataset_name: str) -> Counter[str]:
    counts = Counter(item.label for item in items)
    if set(counts) != {"relevant", "not_relevant"}:
        raise TrainingError(
            f"{dataset_name} must contain relevant and not_relevant labels"
        )
    return counts


def _validate_training_data(items: Sequence[LabeledItem], *, test_size: float) -> None:
    label_counts = _label_counts(items, dataset_name="training data")
    if min(label_counts.values()) < 2:
        raise TrainingError("each label needs at least two items for a stratified split")

    test_count = ceil(len(items) * test_size)
    train_count = len(items) - test_count
    if test_count < len(label_counts) or train_count < len(label_counts):
        raise TrainingError("train/test split is too small to contain both labels")


def _validate_external_holdout(
    training_items: Sequence[LabeledItem],
    evaluation_items: Sequence[LabeledItem],
) -> None:
    _label_counts(training_items, dataset_name="training data")
    _label_counts(evaluation_items, dataset_name="external holdout")
    training_ids = {item.id for item in training_items}
    evaluation_ids = {item.id for item in evaluation_items}
    overlap = sorted(training_ids & evaluation_ids)
    if overlap:
        raise TrainingError(
            "training data and external holdout must be disjoint; overlapping ids: "
            + ", ".join(overlap[:5])
        )


def _split_items(
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int,
    test_size: float,
) -> _Split:
    if not 0.0 < test_size < 1.0:
        raise TrainingError("test_size must be between 0 and 1")
    _validate_training_data(items, test_size=test_size)

    texts = [build_text(item, profile.text_fields) for item in items]
    labels = [item.label for item in items]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    return _Split(
        x_train=list(x_train),
        x_test=list(x_test),
        y_train=list(y_train),
        y_test=list(y_test),
    )


def evaluate_labeled_items(
    model: Pipeline,
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
) -> Metrics:
    """Score a fitted model against an explicit labeled evaluation set."""

    _label_counts(items, dataset_name="evaluation data")
    texts = [build_text(item, profile.text_fields) for item in items]
    labels = [item.label for item in items]
    return calculate_metrics(
        labels,
        model.predict(texts),
        positive_label=profile.positive_label,
    )


def evaluate_trained_model(
    model: Pipeline,
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> Metrics:
    """Recreate the recorded random holdout split and score an already trained model."""

    split = _split_items(
        items, profile, random_seed=random_seed, test_size=test_size
    )
    y_pred = model.predict(split.x_test)
    return calculate_metrics(
        split.y_test,
        y_pred,
        positive_label=profile.positive_label,
    )


def train_and_evaluate(
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """Train on one stratified split and score only the held-out test rows."""

    split = _split_items(
        items, profile, random_seed=random_seed, test_size=test_size
    )
    model = build_pipeline(random_seed=random_seed)
    model.fit(split.x_train, split.y_train)

    # test データはモデルへ見せず、学習後の成績確認だけに使う。
    metrics = calculate_metrics(
        split.y_test,
        model.predict(split.x_test),
        positive_label=profile.positive_label,
    )
    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(split.x_train),
        test_size=len(split.x_test),
        random_seed=random_seed,
    )


def _train_with_external_holdout(
    training_items: Sequence[LabeledItem],
    evaluation_items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int,
) -> TrainingResult:
    """Fit every training row and score only the explicitly separated real holdout."""

    _validate_external_holdout(training_items, evaluation_items)
    model = build_pipeline(random_seed=random_seed)
    model.fit(
        [build_text(item, profile.text_fields) for item in training_items],
        [item.label for item in training_items],
    )
    metrics = evaluate_labeled_items(model, evaluation_items, profile)
    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(training_items),
        test_size=len(evaluation_items),
        random_seed=random_seed,
    )


def _model_hyperparameters(model: Pipeline) -> dict[str, object]:
    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    return {
        "tfidf": {
            "ngram_range": list(tfidf.ngram_range),
            "max_features": tfidf.max_features,
        },
        "classifier": {
            "name": classifier.__class__.__name__,
            "C": float(classifier.C),
            "max_iter": int(classifier.max_iter),
        },
    }


def _created_at(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise TrainingError("created_at must include a timezone")
    return result.astimezone(timezone.utc)


def _model_version(created_at: datetime, data_hash: str) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{data_hash.removeprefix('sha256:')[:8]}"


def train_candidate(
    *,
    items: Sequence[LabeledItem],
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path = "artifacts",
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    evaluation_items: Sequence[LabeledItem] | None = None,
    evaluation_data_path: str | Path | None = None,
    code_commit: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Train, save, reload, evaluate, and describe one candidate artifact."""

    source_items = load_labeled_jsonl(data_path)
    if list(items) != source_items:
        raise TrainingError("items do not match the hashed training data file")
    if (evaluation_items is None) != (evaluation_data_path is None):
        raise TrainingError(
            "evaluation_items and evaluation_data_path must be provided together"
        )

    data_hash = labeled_dataset_hash(data_path)
    timestamp = _created_at(created_at)
    version = _model_version(timestamp, data_hash)

    evaluation_hash: str | None = None
    if evaluation_items is not None and evaluation_data_path is not None:
        source_evaluation_items = load_labeled_jsonl(evaluation_data_path)
        if list(evaluation_items) != source_evaluation_items:
            raise TrainingError(
                "evaluation_items do not match the hashed evaluation data file"
            )
        result = _train_with_external_holdout(
            items,
            evaluation_items,
            profile,
            random_seed=random_seed,
        )
        evaluation_mode = "external_holdout"
        evaluation_hash = labeled_dataset_hash(evaluation_data_path)
        recorded_test_size: float | None = None
    else:
        result = train_and_evaluate(
            items,
            profile,
            random_seed=random_seed,
            test_size=test_size,
        )
        evaluation_mode = "random_holdout"
        recorded_test_size = test_size

    # candidate を一度ディスクへ保存し、再ロードした実物を評価する。保存前の in-memory model だけを信用しない。
    model_path = save_candidate_model(
        result.model,
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    reloaded = load_candidate_model(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    if evaluation_items is not None:
        metrics = evaluate_labeled_items(reloaded, evaluation_items, profile)
    else:
        metrics = evaluate_trained_model(
            reloaded,
            items,
            profile,
            random_seed=random_seed,
            test_size=test_size,
        )
    gate = evaluate_quality_gate(metrics, profile)

    metadata: dict[str, object] = {
        "schema_version": 2,
        "stage": "candidate",
        "profile": profile.profile,
        "model_version": version,
        "training_data_hash": data_hash,
        "evaluation_mode": evaluation_mode,
        "evaluation_data_hash": evaluation_hash,
        "model_artifact_hash": sha256_file(model_path),
        "code_commit": code_commit if code_commit is not None else resolve_code_commit(),
        "random_seed": random_seed,
        "test_size": recorded_test_size,
        "train_size": result.train_size,
        "holdout_size": result.test_size,
        "hyperparameters": _model_hyperparameters(reloaded),
        "metrics": asdict(metrics),
        "quality_gate": asdict(gate),
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "dependencies": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_candidate_metadata(
        metadata,
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    return metadata


def evaluate_candidate(
    *,
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
    evaluation_data_path: str | Path | None = None,
) -> tuple[Metrics, dict[str, object]]:
    """Re-evaluate a saved candidate using its recorded evaluation contract."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    current_hash = labeled_dataset_hash(data_path)
    if metadata.get("training_data_hash") != current_hash:
        raise ArtifactError(
            "training data hash changed; this candidate cannot be re-evaluated against a different dataset"
        )

    items = load_labeled_jsonl(data_path)
    model = load_candidate_model(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    evaluation_mode = str(metadata.get("evaluation_mode") or "random_holdout")
    if evaluation_mode == "external_holdout":
        if evaluation_data_path is None:
            raise ArtifactError(
                "external-holdout candidate requires evaluation_data_path"
            )
        current_evaluation_hash = labeled_dataset_hash(evaluation_data_path)
        if metadata.get("evaluation_data_hash") != current_evaluation_hash:
            raise ArtifactError(
                "evaluation data hash changed; this candidate cannot be re-evaluated against a different holdout"
            )
        evaluation_items = load_labeled_jsonl(evaluation_data_path)
        _validate_external_holdout(items, evaluation_items)
        metrics = evaluate_labeled_items(model, evaluation_items, profile)
    elif evaluation_mode == "random_holdout":
        raw_test_size = metadata.get("test_size")
        if not isinstance(raw_test_size, (int, float)):
            raise ArtifactError("random-holdout candidate is missing test_size")
        metrics = evaluate_trained_model(
            model,
            items,
            profile,
            random_seed=int(metadata["random_seed"]),
            test_size=float(raw_test_size),
        )
    else:
        raise ArtifactError(f"unsupported evaluation_mode: {evaluation_mode}")

    gate = evaluate_quality_gate(metrics, profile)
    evaluation = {
        "profile": profile.profile,
        "model_version": model_version,
        "evaluation_mode": evaluation_mode,
        "evaluation_data_hash": metadata.get("evaluation_data_hash"),
        "metrics": asdict(metrics),
        "quality_gate": asdict(gate),
    }
    return metrics, evaluation
