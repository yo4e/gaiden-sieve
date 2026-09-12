"""Minimal local CLI for Phase 1 training and classification."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from gaiden_sieve.classify import predict_one
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.train import train_and_evaluate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gaiden_sieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "classify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--profile", required=True)
        subparser.add_argument(
            "--profiles-dir",
            default="profiles",
            help="directory containing profile folders (default: profiles)",
        )

    classify = subparsers.choices["classify"]
    classify.add_argument("--title", required=True)
    classify.add_argument("--summary", default="")
    return parser


def _load_phase1_inputs(profile_name: str, profiles_dir: str | Path):
    profile_dir = Path(profiles_dir) / profile_name
    profile = load_profile(profile_dir / "config.yml")
    if profile.profile != profile_name:
        raise ValueError(
            f"profile directory '{profile_name}' contains config for '{profile.profile}'"
        )
    items = load_labeled_jsonl(profile_dir / "labeled.jsonl")
    return profile, items


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile, items = _load_phase1_inputs(args.profile, args.profiles_dir)
    result = train_and_evaluate(items, profile)

    payload: dict[str, object] = {
        "profile": profile.profile,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "random_seed": result.random_seed,
        "metrics": asdict(result.metrics),
    }

    if args.command == "classify":
        prediction = predict_one(
            result.model,
            profile,
            title=args.title,
            summary=args.summary,
        )
        payload["prediction"] = asdict(prediction)

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
