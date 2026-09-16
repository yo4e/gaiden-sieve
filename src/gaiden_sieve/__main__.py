"""Local CLI for the GAIDEN SIEVE MLOps loop."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gaiden_sieve.artifacts import latest_candidate_version, load_production
from gaiden_sieve.batch import (
    classify_batch,
    load_input_jsonl,
    uncertain_rows,
    write_classified_jsonl,
    write_uncertain_queue,
)
from gaiden_sieve.classify import predict_one
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.drift import build_drift_report, write_drift_report
from gaiden_sieve.profile import load_profile
from gaiden_sieve.promote import promote_candidate
from gaiden_sieve.reproducibility import verify_candidate_reproducibility
from gaiden_sieve.shadow import run_shadow_candidate
from gaiden_sieve.train import evaluate_candidate, train_candidate


def _add_common_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--profile", required=True)
    subparser.add_argument(
        "--profiles-dir",
        default="profiles",
        help="directory containing profile folders (default: profiles)",
    )
    subparser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="directory containing model artifacts (default: artifacts)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gaiden_sieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "verify", "promote", "classify", "drift", "shadow"):
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)

    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument(
        "--candidate",
        help="candidate model_version; default: newest candidate metadata",
    )

    verify = subparsers.choices["verify"]
    verify.add_argument(
        "--candidate",
        help="candidate model_version; default: newest candidate metadata",
    )

    promote = subparsers.choices["promote"]
    promote.add_argument(
        "--candidate",
        required=True,
        help="candidate model_version to promote explicitly",
    )

    classify = subparsers.choices["classify"]
    classify.add_argument("--title")
    classify.add_argument("--summary", default="")
    classify.add_argument("--input", help="JSONL batch to classify with production")
    classify.add_argument("--output", help="classified JSONL output path")
    classify.add_argument(
        "--uncertain-output",
        help="optional uncertain-only JSONL queue path",
    )

    drift = subparsers.choices["drift"]
    drift.add_argument("--input", required=True, help="JSONL batch to observe")
    drift.add_argument("--output", help="drift report JSON path")
    drift.add_argument(
        "--uncertain-output",
        help="uncertain queue path; default: reports/<profile>/uncertain.jsonl",
    )
    drift.add_argument(
        "--reports-dir",
        default="reports",
        help="base directory for default drift outputs (default: reports)",
    )

    shadow = subparsers.choices["shadow"]
    shadow.add_argument("--input", required=True, help="AI外電 shadow input JSONL")
    shadow.add_argument(
        "--candidate",
        help="gate-passing candidate model_version; default: newest candidate metadata",
    )
    shadow.add_argument("--comparison-output")
    shadow.add_argument("--classified-output")
    shadow.add_argument("--drift-output")
    shadow.add_argument("--uncertain-output")
    shadow.add_argument(
        "--reports-dir",
        default="reports",
        help="base directory for default shadow outputs (default: reports)",
    )
    return parser


def _load_profile_inputs(profile_name: str, profiles_dir: str | Path):
    profile_dir = Path(profiles_dir) / profile_name
    profile = load_profile(profile_dir / "config.yml")
    if profile.profile != profile_name:
        raise ValueError(
            f"profile directory '{profile_name}' contains config for '{profile.profile}'"
        )
    return profile, profile_dir / "labeled.jsonl"


def _production_batch(args, profile):
    model, metadata = load_production(
        artifacts_dir=args.artifacts_dir,
        profile=profile.profile,
    )
    items = load_input_jsonl(args.input)
    classified_at = datetime.now(timezone.utc)
    rows = classify_batch(
        model,
        profile,
        items,
        model_version=str(metadata["model_version"]),
        classified_at=classified_at,
    )
    return model, metadata, items, rows, classified_at


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile, data_path = _load_profile_inputs(args.profile, args.profiles_dir)
    exit_code = 0

    if args.command == "train":
        items = load_labeled_jsonl(data_path)
        payload = train_candidate(
            items=items,
            data_path=data_path,
            profile=profile,
            artifacts_dir=args.artifacts_dir,
        )
    elif args.command == "evaluate":
        model_version = args.candidate or latest_candidate_version(
            artifacts_dir=args.artifacts_dir,
            profile=profile.profile,
        )
        _, payload = evaluate_candidate(
            data_path=data_path,
            profile=profile,
            artifacts_dir=args.artifacts_dir,
            model_version=model_version,
        )
    elif args.command == "verify":
        model_version = args.candidate or latest_candidate_version(
            artifacts_dir=args.artifacts_dir,
            profile=profile.profile,
        )
        payload = verify_candidate_reproducibility(
            data_path=data_path,
            profile=profile,
            artifacts_dir=args.artifacts_dir,
            model_version=model_version,
        )
        exit_code = 0 if payload["passed"] else 2
    elif args.command == "promote":
        payload = promote_candidate(
            profile=profile,
            artifacts_dir=args.artifacts_dir,
            model_version=args.candidate,
        )
    elif args.command == "classify":
        if args.input:
            if args.title:
                raise ValueError("--title cannot be combined with --input")
            if not args.output:
                raise ValueError("--output is required when --input is used")
            _, metadata, _, rows, _ = _production_batch(args, profile)
            write_classified_jsonl(args.output, rows)
            if args.uncertain_output:
                write_uncertain_queue(args.uncertain_output, rows)
            payload = {
                "profile": profile.profile,
                "production_model_version": metadata["model_version"],
                "input_count": len(rows),
                "uncertain_count": len(uncertain_rows(rows)),
                "classified_output": str(Path(args.output)),
                "uncertain_output": (
                    str(Path(args.uncertain_output)) if args.uncertain_output else None
                ),
            }
        else:
            if not args.title:
                raise ValueError("--title is required unless --input is used")
            model, metadata = load_production(
                artifacts_dir=args.artifacts_dir,
                profile=profile.profile,
            )
            prediction = predict_one(
                model,
                profile,
                title=args.title,
                summary=args.summary,
            )
            payload = {
                "profile": profile.profile,
                "production_model_version": metadata["model_version"],
                "prediction": asdict(prediction),
            }
    elif args.command == "drift":
        model, metadata, items, rows, generated_at = _production_batch(args, profile)
        report = build_drift_report(
            model,
            profile,
            items,
            rows,
            model_version=str(metadata["model_version"]),
            generated_at=generated_at,
        )
        stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        report_path = (
            Path(args.output)
            if args.output
            else Path(args.reports_dir) / profile.profile / f"drift-{stamp}.json"
        )
        queue_path = (
            Path(args.uncertain_output)
            if args.uncertain_output
            else Path(args.reports_dir) / profile.profile / "uncertain.jsonl"
        )
        write_drift_report(report_path, report)
        write_uncertain_queue(queue_path, rows)
        payload = {
            "report": report,
            "report_path": str(report_path),
            "uncertain_count": len(uncertain_rows(rows)),
            "uncertain_queue_path": str(queue_path),
        }

    else:
        model_version = args.candidate or latest_candidate_version(
            artifacts_dir=args.artifacts_dir,
            profile=profile.profile,
        )
        generated_at = datetime.now(timezone.utc)
        stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        report_dir = Path(args.reports_dir) / profile.profile
        payload = run_shadow_candidate(
            profile=profile,
            artifacts_dir=args.artifacts_dir,
            model_version=model_version,
            input_path=args.input,
            comparison_output=(
                Path(args.comparison_output)
                if args.comparison_output
                else report_dir / f"shadow-comparison-{stamp}.json"
            ),
            classified_output=(
                Path(args.classified_output)
                if args.classified_output
                else report_dir / f"shadow-classified-{stamp}.jsonl"
            ),
            drift_output=(
                Path(args.drift_output)
                if args.drift_output
                else report_dir / f"shadow-drift-{stamp}.json"
            ),
            uncertain_output=(
                Path(args.uncertain_output)
                if args.uncertain_output
                else report_dir / f"shadow-uncertain-{stamp}.jsonl"
            ),
            generated_at=generated_at,
        )


    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
