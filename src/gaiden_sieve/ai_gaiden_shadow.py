"""Read AI外電 official-source inputs for a side-effect-free shadow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class AiGaidenShadowError(RuntimeError):
    """Raised when AI外電 inputs cannot be exported safely."""


def _item_key(item: Any) -> tuple[str, str]:
    return (
        str(getattr(item, "dedupe_key", "")),
        str(getattr(item, "canonical_url", "")),
    )


def _stable_id(item: Any) -> str:
    source = str(item.source_id)
    dedupe_key = str(getattr(item, "dedupe_key", ""))
    if dedupe_key:
        return f"{source}:{dedupe_key}"
    fallback = f"{source}\n{item.canonical_url}\n{item.title}".encode("utf-8")
    return f"{source}:shadow:{hashlib.sha256(fallback).hexdigest()}"


def _published_at(item: Any) -> str | None:
    value = getattr(item, "published_at", None)
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_shadow_rows(
    raw_results: Sequence[Any],
    admitted_results: Sequence[Any],
    rules: Mapping[str, Any],
    *,
    ai_gaiden_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Join pre-admission inputs with the existing admission outcome."""

    admitted_by_source = {result.config.id: result for result in admitted_results}
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        if not result.success or result.not_modified:
            continue
        admitted = admitted_by_source.get(result.config.id)
        accepted_keys = (
            {_item_key(item) for item in admitted.items}
            if admitted is not None and admitted.success
            else set()
        )
        rule = rules.get(result.config.id)
        mode = str(getattr(rule, "mode", "all"))
        policy = getattr(rule, "policy", None)

        for item in result.items:
            rows.append(
                {
                    "id": _stable_id(item),
                    "source": item.source_id,
                    "title": item.title,
                    "summary": item.summary,
                    "source_url": item.canonical_url,
                    "published_at": _published_at(item),
                    "admission_mode": mode,
                    "admission_policy": policy,
                    "admission_accepted": _item_key(item) in accepted_keys,
                    "ai_gaiden_commit": ai_gaiden_commit,
                }
            )
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return output_path


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def collect_ai_gaiden_shadow(
    *,
    ai_gaiden_root: str | Path,
    output_path: str | Path,
    cache_path: str | Path,
) -> dict[str, Any]:
    """Fetch current AI外電 official sources and export both sides of admission."""

    root = Path(ai_gaiden_root).resolve()
    if not (root / "config" / "feeds.yml").exists():
        raise AiGaidenShadowError(f"AI外電 checkout not found: {root}")

    # 既存の取得・正規化・admission 実装をそのまま使い、shadow 側でロジックを複製しない。
    sys.path.insert(0, str(root))
    try:
        from scripts.admission import (  # type: ignore[import-not-found]
            apply_admission,
            expand_configs_for_admission,
            load_admission_config,
        )
        from scripts.source_reader import SourceReader  # type: ignore[import-not-found]
        from scripts.utils import load_feed_configs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AiGaidenShadowError(
            "AI外電 integration dependencies are not installed"
        ) from exc
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)

    configs = load_feed_configs(root / "config" / "feeds.yml")
    rules = load_admission_config(root / "config" / "admission.yml")
    config_by_id = {config.id: config for config in configs}
    expanded = expand_configs_for_admission(configs, rules)

    reader = SourceReader(Path(cache_path))
    # workflow_dispatch のたびに現在の候補を観察する。304 によって空観測になるのを避ける。
    reader.use_conditional_requests = False
    raw_results = reader.fetch_all(expanded)
    admitted_results = apply_admission(raw_results, rules, config_by_id)
    commit = _git_commit(root)
    rows = build_shadow_rows(
        raw_results,
        admitted_results,
        rules,
        ai_gaiden_commit=commit,
    )
    if not rows:
        failed = [result.config.id for result in raw_results if not result.success]
        raise AiGaidenShadowError(
            "AI外電 returned no shadow inputs"
            + (f"; failed sources: {', '.join(failed)}" if failed else "")
        )
    _write_jsonl(output_path, rows)

    accepted = sum(1 for row in rows if row["admission_accepted"])
    failures = [
        {"source": result.config.id, "error": result.error}
        for result in raw_results
        if not result.success
    ]
    return {
        "ai_gaiden_commit": commit,
        "configured_sources": len([config for config in configs if config.enabled]),
        "successful_sources": len([result for result in raw_results if result.success]),
        "failed_sources": failures,
        "input_count": len(rows),
        "admission_accepted": accepted,
        "admission_rejected": len(rows) - accepted,
        "output": str(Path(output_path)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export live AI外電 inputs for GAIDEN SIEVE shadow observation"
    )
    parser.add_argument("--ai-gaiden-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".shadow/ai-gaiden-source-state.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = collect_ai_gaiden_shadow(
        ai_gaiden_root=args.ai_gaiden_root,
        output_path=args.output,
        cache_path=args.cache,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
