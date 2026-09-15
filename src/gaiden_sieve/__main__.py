"""Local CLI for the Phase 2 model lifecycle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from gaiden_sieve.artifacts import latest_candidate_version, load_production
from gaiden_sieve.classify import predict_one
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.promote import promote_candidate
from gaiden_sieve.train import evaluate_candidate, train_candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gaiden_sieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "promote", "classify"):
        subparser = subparsers.add_parser(command)
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

    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument(
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
    classify.add_argument("--title", required=True)
    classify.add_argument("--summary", default="")
    return parser


def _load_profile_inputs(profile_name: str, profiles_dir: str | Path):
    profile_dir = Path(profiles_dir) / profile_name
    profile = load_profile(profile_dir / "config.yml")
    if profile.profile != profile_name:
        raise ValueError(
            f"profile directory '{profile_name}' contains config for '{profile.profile}'"
        )
    return profile, profile_dir / "labeled.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile, data_path = _load_profile_inputs(args.profile, args.profiles_dir)

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
    elif args.command == "promote":
        payload = promote_candidate(
            profile=profile,
            artifacts_dir=args.artifacts_dir,
            model_version=args.candidate,
        )
    else:
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

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
