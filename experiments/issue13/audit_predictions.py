"""保存済み予測の独立監査。学習・取得・しきい値変更は行わない。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

CANDIDATES = ("baseline", "iteration1", "iteration2")
POSITIVE = "relevant"
NEGATIVE = "not_relevant"
SOURCEGRAPH = "sourcegraph-changelog"


def operational_label(probability: float) -> str:
    if probability >= 0.80:
        return POSITIVE
    if probability < 0.40:
        return NEGATIVE
    return "uncertain"


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("a scored population must not be empty")
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("ids must be nonempty and unique within each population")
        seen.add(item_id)
        if row.get("label") not in {POSITIVE, NEGATIVE}:
            raise ValueError("unsupported label")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError("source must be a nonempty string")
        p = row.get("probability_relevant")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError("probability must be numeric, not boolean")
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        # 二値モデルの同点は classes_[0] = not_relevant。運用の >=0.80 とは別。
        binary = POSITIVE if p > 0.50 else NEGATIVE
        if "binary_prediction" in row and row["binary_prediction"] != binary:
            raise ValueError("stored binary prediction disagrees with probability")
        if "classification" in row and row["classification"] != operational_label(p):
            raise ValueError("stored operational label disagrees with probability")


def decision_metrics(labels: list[str], selected: list[bool]) -> dict[str, Any]:
    if len(labels) != len(selected):
        raise ValueError("labels and decisions must have equal lengths")
    tp = sum(y == POSITIVE and take for y, take in zip(labels, selected))
    fp = sum(y == NEGATIVE and take for y, take in zip(labels, selected))
    fn = sum(y == POSITIVE and not take for y, take in zip(labels, selected))
    tn = sum(y == NEGATIVE and not take for y, take in zip(labels, selected))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "selected_count": tp + fp,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rows(rows)
    labels = [row["label"] for row in rows]
    scores = [row["probability_relevant"] for row in rows]
    positives = [p for y, p in zip(labels, scores) if y == POSITIVE]
    negatives = [p for y, p in zip(labels, scores) if y == NEGATIVE]
    pair_count = len(positives) * len(negatives)
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    # 全境界での実現可能性の検査のみ。最適なしきい値を選択・保存・適用しない。
    attainable = []
    for boundary in sorted({0.0, 1.0, *scores}):
        metrics = decision_metrics(labels, [p >= boundary for p in scores])
        if metrics["recall"] is not None and metrics["recall"] >= 0.80:
            attainable.append(metrics["precision"])
    best_precision = max((p for p in attainable if p is not None), default=None)
    source_control = [row["source"] != SOURCEGRAPH for row in rows]
    distribution = Counter(operational_label(p) for p in scores)
    return {
        "count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "binary": decision_metrics(labels, [p > 0.50 for p in scores]),
        "operational_relevant": decision_metrics(labels, [p >= 0.80 for p in scores]),
        "retained_for_review": decision_metrics(labels, [p >= 0.40 for p in scores]),
        "operational_distribution": {key: distribution[key] for key in (POSITIVE, "uncertain", NEGATIVE)},
        "automatic_decision_rate": (distribution[POSITIVE] + distribution[NEGATIVE]) / len(rows),
        "positive_score_min": min(positives) if positives else None,
        "positive_score_max": max(positives) if positives else None,
        "negative_score_min": min(negatives) if negatives else None,
        "negative_score_max": max(negatives) if negatives else None,
        "separation_gap": min(positives) - max(negatives) if pair_count else None,
        "pair_count": pair_count,
        "positive_wins": wins,
        "ties": ties,
        "roc_auc": (wins + 0.5 * ties) / pair_count if pair_count else None,
        "threshold_feasibility_audit_only": {
            "precision_target": 0.95, "recall_target": 0.80,
            "feasible": best_precision is not None and best_precision >= 0.95,
            "best_precision_at_recall_target": best_precision,
        },
        # 悪い比較器でも試験を通れるかの反証用。モデルの実装や採用案ではない。
        "source_only_negative_control": decision_metrics(labels, source_control),
        "source_control_binary_agreement": sum(
            take == (p > 0.50) for take, p in zip(source_control, scores)
        ) / len(rows),
    }


def population_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_sourcegraph = [row for row in rows if row["source"] != SOURCEGRAPH]
    return {
        "all": summarize(rows),
        "non_sourcegraph": summarize(non_sourcegraph) if non_sourcegraph else None,
    }


def audit_document(document: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "scope": "saved predictions only; no fitting, acquisition, tuning or promotion",
        "gold": {},
    }
    for candidate in CANDIDATES:
        rows = [
            dict(id=row["id"], source=row["source"], label=row["label"], **row[candidate])
            for row in document["gold"]
        ]
        report["gold"][candidate] = population_report(rows)
    report["iteration2_training"] = population_report(document["iteration2_training"])
    train_ids = {row["id"] for row in document["iteration2_training"]}
    gold_ids = {row["id"] for row in document["gold"]}
    report["train_gold_id_overlap_count"] = len(train_ids & gold_ids)
    if train_ids & gold_ids:
        raise ValueError("training and gold ids overlap")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path(__file__).with_name("prediction-comparison.json"))
    args = parser.parse_args()
    try:
        raw = args.predictions.read_bytes()
        result = audit_document(json.loads(raw))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"audit failed: {exc}\n")
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
