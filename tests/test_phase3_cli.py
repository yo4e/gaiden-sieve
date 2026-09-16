import json
from pathlib import Path

from gaiden_sieve.__main__ import main


ROOT = Path(__file__).resolve().parents[1]


def test_phase3_cli_batch_drift_queue_and_reproducibility(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    common = [
        "--profile",
        "ai-gaiden",
        "--profiles-dir",
        str(ROOT / "profiles"),
        "--artifacts-dir",
        str(artifacts),
    ]

    assert main(["train", *common]) == 0
    trained = json.loads(capsys.readouterr().out)
    version = trained["model_version"]

    assert main(["verify", *common, "--candidate", version]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["passed"] is True

    assert main(["promote", *common, "--candidate", version]) == 0
    capsys.readouterr()

    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "incoming-1",
                        "source": "fixture-a",
                        "title": "New AI reasoning model released",
                        "summary": "The model uses tools and plans tasks.",
                    }
                ),
                json.dumps(
                    {
                        "id": "incoming-2",
                        "source": "fixture-b",
                        "title": "Football club signs a striker",
                        "summary": "The transfer was announced today.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    classified = tmp_path / "classified.jsonl"
    queue = tmp_path / "uncertain.jsonl"
    assert main(
        [
            "classify",
            *common,
            "--input",
            str(incoming),
            "--output",
            str(classified),
            "--uncertain-output",
            str(queue),
        ]
    ) == 0
    batch_payload = json.loads(capsys.readouterr().out)
    assert batch_payload["input_count"] == 2
    assert len(classified.read_text(encoding="utf-8").splitlines()) == 2
    assert queue.exists()

    report_path = tmp_path / "drift.json"
    drift_queue = tmp_path / "drift-uncertain.jsonl"
    assert main(
        [
            "drift",
            *common,
            "--input",
            str(incoming),
            "--output",
            str(report_path),
            "--uncertain-output",
            str(drift_queue),
        ]
    ) == 0
    drift_payload = json.loads(capsys.readouterr().out)
    assert drift_payload["report"]["input_count"] == 2
    assert report_path.exists()
    assert drift_queue.exists()
