"""Record source identity and actual execution metadata, never individual rows."""

from dataclasses import asdict
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
from platform import python_version
import subprocess
from tempfile import TemporaryDirectory

import mlflow

from credit_risk import TrainingConfig, TrainingResult

MODEL_DEPENDENCIES = ("mlflow", "scikit-learn", "numpy", "pandas", "cloudpickle", "matplotlib")


def _git_metadata() -> dict[str, str | bool | None]:
    unavailable: dict[str, str | bool | None] = {"status": "unavailable", "commit": None, "dirty": None}
    root = Path(__file__).resolve().parent.parent
    # A caller's shell may target a different repository or index.
    environment = {key: value for key, value in os.environ.items() if key not in {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
    }}

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments], env=environment,
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return unavailable  # Never attribute an enclosing repository to this copy.
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return unavailable
    return {"status": "available", "commit": commit, "dirty": dirty}


def input_fingerprints(config: TrainingConfig) -> dict[str, str]:
    fingerprints = {}
    for name, path in [
        ("features_sha256", config.feature_path),
        ("payment_status_sha256", config.payment_status_path),
    ]:
        with path.open("rb") as source:
            fingerprints[name] = file_digest(source, "sha256").hexdigest()
    return fingerprints


def log_provenance(result: TrainingResult, fingerprints: dict[str, str]) -> None:
    data = {
        **fingerprints, **asdict(result.data_quality),
        "retained_rows": result.data_quality.retained_rows,
        "training_rows": result.training_rows, "test_rows": result.test_rows,
    }
    software = {
        "python": python_version(),
        **{name: version(name) for name in (*MODEL_DEPENDENCIES, "scipy", "joblib", "threadpoolctl")},
    }
    git = _git_metadata()
    mlflow.set_tags({
        **{f"data.{name}": str(value) for name, value in data.items()},
        **{f"software.{name}": value for name, value in software.items()},
        **{f"git.{name}": str(value).lower() if value is not None else "unavailable"
           for name, value in git.items()},
    })
    with TemporaryDirectory(prefix="credit-risk-provenance-") as temporary:
        metadata = Path(temporary) / "metadata.json"
        metadata.write_text(
            json.dumps({"data": data, "software": software, "git": git}, indent=2), encoding="utf-8",
        )
        mlflow.log_artifact(str(metadata), artifact_path="provenance")
