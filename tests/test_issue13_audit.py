"""運用境界と順位を混同しないための、学習を伴わない独立監査テスト。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "issue13"
SPEC = importlib.util.spec_from_file_location("issue13_audit", HERE / "audit_predictions.py")
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def row(item_id, label, score, source="example"):
    return dict(id=item_id, label=label, probability_relevant=score, source=source)


@pytest.mark.parametrize("probability, expected", [
    (0.0, "not_relevant"), (0.399999, "not_relevant"),
    (0.4, "uncertain"), (0.5, "uncertain"),
    (0.799999, "uncertain"), (0.8, "relevant"), (1.0, "relevant"),
])
def test_operational_boundaries(probability, expected):
    assert AUDIT.operational_label(probability) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.7"])
def test_invalid_probability_is_rejected(bad):
    with pytest.raises(ValueError):
        AUDIT.summarize([row("p", "relevant", bad)])


def test_zero_predictions_are_undefined_not_zero_precision():
    result = AUDIT.summarize([row("p", "relevant", .7), row("n", "not_relevant", .6)])
    assert result["operational_relevant"]["precision"] is None
    assert result["operational_relevant"]["recall"] == 0
    assert result["retained_for_review"]["recall"] == 1
    assert result["roc_auc"] == 1
    assert result["binary"]["fp"] == 1


def test_auc_ties_and_single_class():
    result = AUDIT.summarize([row("p", "relevant", .5), row("n", "not_relevant", .5)])
    assert result["roc_auc"] == .5
    assert result["binary"]["selected_count"] == 0
    assert not result["threshold_feasibility_audit_only"]["feasible"]
    assert AUDIT.summarize([row("p", "relevant", .7)])["roc_auc"] is None
    assert AUDIT.summarize([row("n", "not_relevant", .7)])["binary"]["recall"] is None


def test_duplicate_ids_and_inconsistent_saved_predictions():
    with pytest.raises(ValueError):
        AUDIT.summarize([row("p", "relevant", .7)] * 2)
    for field, value in [("classification", "relevant"), ("binary_prediction", "not_relevant")]:
        with pytest.raises(ValueError):
            AUDIT.summarize([dict(row("p", "relevant", .7), **{field: value})])


def test_source_control_does_not_claim_model_uses_source():
    result = AUDIT.summarize([
        row("p", "relevant", .6),
        row("hard-n", "not_relevant", .55),
        row("easy-n", "not_relevant", .2, AUDIT.SOURCEGRAPH),
    ])
    assert result["source_control_binary_agreement"] == 1
    assert result["source_only_negative_control"]["precision"] == .5
    assert result["roc_auc"] == 1


def test_committed_prediction_evidence():
    # 原実験を再学習せず、Git管理済みの予測値そのものを検算する。
    document = json.loads((HERE / "prediction-comparison.json").read_text(encoding="utf-8"))
    result = AUDIT.audit_document(document)
    train = result["iteration2_training"]["all"]
    assert train["count"] == 59
    assert train["positive_count"] == 38 and train["negative_count"] == 21
    assert train["roc_auc"] == 1.0
    assert train["positive_score_min"] == pytest.approx(.7505572839082452)
    assert train["negative_score_max"] == pytest.approx(.6195050841792884)
    assert train["separation_gap"] == pytest.approx(.13105219972895676)
    assert train["binary"]["fp"] == 9
    assert train["operational_distribution"] == dict(relevant=0, uncertain=47, not_relevant=12)
    assert result["train_gold_id_overlap_count"] == 0
    for candidate, expected_auc, expected_precision in [
        ("baseline", .625, .8), ("iteration1", .625, .8), ("iteration2", .375, 2 / 3),
    ]:
        gold = result["gold"][candidate]
        assert gold["non_sourcegraph"]["count"] == 6
        assert gold["non_sourcegraph"]["roc_auc"] == expected_auc
        assert gold["all"]["binary"]["precision"] == pytest.approx(2 / 3)
        feasibility = gold["all"]["threshold_feasibility_audit_only"]
        assert feasibility["best_precision_at_recall_target"] == pytest.approx(expected_precision)
        assert not feasibility["feasible"]
    bad = copy.deepcopy(document)
    bad["iteration2_training"][0]["id"] = bad["gold"][0]["id"]
    with pytest.raises(ValueError, match="overlap"):
        AUDIT.audit_document(bad)


def test_committed_feature_aliases_and_weak_holdout_control():
    from gaiden_sieve.data import build_text, load_labeled_jsonl
    from gaiden_sieve.profile import load_profile
    from gaiden_sieve.train import build_pipeline

    profile = load_profile(ROOT / "profiles/ai-gaiden/config.yml")
    training = load_labeled_jsonl(ROOT / "profiles/ai-gaiden/bootstrap")
    holdout = load_labeled_jsonl(ROOT / "profiles/ai-gaiden/real-holdout")
    analyzer = build_pipeline().named_steps["tfidf"].build_analyzer()
    train_family = [r for r in training if r.source == AUDIT.SOURCEGRAPH and not r.summary]
    holdout_family = [r for r in holdout if r.source == AUDIT.SOURCEGRAPH and not r.summary]
    assert len(train_family) == 7 and len(holdout_family) == 2
    # 単なる近似ではなく、実際のモデル用analyzer出力が完全一致する。
    fingerprints = {tuple(analyzer(build_text(r, profile.text_fields))) for r in train_family + holdout_family}
    assert len(fingerprints) == 1
    assert not ({r.id for r in training} & {r.id for r in holdout})
    control = AUDIT.decision_metrics(
        [r.label for r in holdout], [r.source != AUDIT.SOURCEGRAPH for r in holdout]
    )
    assert len(holdout) == 10
    assert control["tp"] == 6 and control["tn"] == 4
    assert control["precision"] == control["recall"] == 1.0
