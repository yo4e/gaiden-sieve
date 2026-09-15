import json
from pathlib import Path
from gaiden_sieve.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def test_phase2_cli_runs_explicit_lifecycle(tmp_path: Path, capsys) -> None:
    common = [
        "--profile", "ai-gaiden",
        "--profiles-dir", str(ROOT / "profiles"),
        "--artifacts-dir", str(tmp_path),
    ]
    assert main(["train", *common]) == 0
    trained = json.loads(capsys.readouterr().out)
    version = trained["model_version"]
    assert trained["stage"] == "candidate"

    assert main(["evaluate", *common, "--candidate", version]) == 0
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["quality_gate"]["passed"] is True

    assert main(["promote", *common, "--candidate", version]) == 0
    promoted = json.loads(capsys.readouterr().out)
    assert promoted["stage"] == "production"

    assert main([
        "classify", *common,
        "--title", "New AI agent model announced",
        "--summary", "The system plans tasks and uses external tools.",
    ]) == 0
    classified = json.loads(capsys.readouterr().out)
    assert classified["production_model_version"] == version
    assert classified["prediction"]["classification"] in {"relevant", "uncertain", "not_relevant"}
