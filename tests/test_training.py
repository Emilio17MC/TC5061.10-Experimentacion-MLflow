from pathlib import Path
from dataclasses import FrozenInstanceError, replace
import json
import os
import shutil
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from credit_risk import TrainingConfig, load_historical_loan_sample, run_training


@pytest.fixture
def loan_files(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    # Two separable groups with both payment outcomes in the holdout.
    payment_status = np.tile([0, 1], 30)
    features = pd.DataFrame({
        "funded_amnt": 2000 + payment_status * 8000 + np.arange(60),
        "interest_rate": 5 + payment_status * 20,
        "annual_income": 80000 - payment_status * 50000,
        "dept_paym_income_ratio": 5 + payment_status * 25,
        "num_30+_delinq_in_2yrs": payment_status,
        "num_inq_in_6mths": payment_status,
        "num_open_credit_lines": 5,
        "num_derogatory_pub_rec": payment_status,
        "total_credit_revolving_bal": 1000,
        "used_credit_share": 20,
        "tot_num_credit_lines": 10,
        "loan_term_months": 36,
        "emp_length": 5,
        "grade": np.where(payment_status == 0, "A1", "G5"),
        "home_ownership_status": "RENT",
        "verification_status": "Verified",
        "loan_purpose": "debt_consolidation",
    })
    feature_path, status_path = tmp_path / "X.csv", tmp_path / "target.csv"
    features.to_csv(feature_path, index=False)
    pd.DataFrame({"y": payment_status}).to_csv(status_path, index=False)
    return feature_path, status_path, features


@pytest.fixture
def training_config(loan_files) -> TrainingConfig:
    feature_path, status_path, _ = loan_files
    return TrainingConfig(
        feature_path=feature_path, payment_status_path=status_path,
        figure_path=feature_path.parent / "evaluation.png",
    )


@pytest.fixture
def local_tracking(tmp_path, monkeypatch):
    from credit_risk import config

    # Substitute only the filesystem location for in-process executable tests.
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path / ".mlflow"


@pytest.fixture
def cli_project(tmp_path):
    project = tmp_path / "checkout"
    project.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "credit_risk", project / "credit_risk")
    shutil.copy2(source / "train.py", project / "train.py")
    shutil.copy2(source / "Makefile", project / "Makefile")
    return project


def run_cli(project, *arguments, runner="python"):
    command = [sys.executable, str(project / "train.py"), *arguments]
    if runner == "make":
        command = [
            "make", "-f", str(project / "Makefile"), "train",
            f"PYTHON={sys.executable}", f"ARGS={shlex.join(arguments)}",
        ]
    return subprocess.run(
        command,
        cwd=project.parent, capture_output=True, text=True, timeout=30,
    )


def copy_cli_inputs(project, feature_path, status_path):
    data = project / "data"
    data.mkdir()
    shutil.copy2(feature_path, data / "X.csv")
    shutil.copy2(status_path, data / "target.csv")


def cli_model_uri(process):
    return next(line.removeprefix("Modelo MLflow: ") for line in process.stdout.splitlines()
                if line.startswith("Modelo MLflow: "))


def test_cli_selects_tracking_store_and_experiment(cli_project, loan_files):
    from mlflow import MlflowClient
    import mlflow
    import mlflow.sklearn

    copy_cli_inputs(cli_project, *loan_files[:2])
    store = cli_project.parent / "selected store.db"
    uri = f"sqlite:///{store}"
    process = run_cli(cli_project, "--tracking-uri", uri, "--experiment-name", "Prueba distinta")
    assert process.returncode == 0, process.stderr
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("Prueba distinta")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    assert len(run.data.metrics) == 12
    assert Path(client.download_artifacts(run.info.run_id, "evaluation/metrics.json")).is_file()
    assert Path(client.download_artifacts(run.info.run_id, "evaluation/evaluation_baseline.png")).stat().st_size > 0
    assert Path(client.download_artifacts(run.info.run_id, "provenance/metadata.json")).is_file()
    mlflow.set_tracking_uri(uri)
    model = mlflow.sklearn.load_model(cli_model_uri(process))
    assert model.predict(loan_files[2].iloc[:4]).tolist() == [0, 1, 0, 1]
    assert not (cli_project / ".mlflow").exists()
    assert not (cli_project.parent / "mlruns").exists()


@pytest.mark.parametrize("option", ["--experiment-name", "--tracking-uri"])
@pytest.mark.parametrize("value", ["", "   "])
def test_cli_rejects_empty_tracking_options_before_side_effects(cli_project, option, value):
    process = run_cli(cli_project, option, value)
    assert process.returncode == 2
    assert "no vacío" in process.stderr
    assert "Traceback" not in process.stderr
    assert not process.stdout
    assert not (cli_project / ".mlflow").exists()
    assert not (cli_project.parent / "mlruns").exists()
    assert not (cli_project / "reports").exists()


def test_cli_alternates_destinations_and_groups_experiments(cli_project, loan_files):
    from mlflow import MlflowClient

    copy_cli_inputs(cli_project, *loan_files[:2])
    # Relative SQLite locations use the invoker's directory, just like CSV paths.
    stores = [cli_project.parent / "first.db", cli_project.parent / "second.db"]
    for uri, name, seed, runner in [
        (f"sqlite:///{stores[0]}", "Comparación", 17, "python"),
        ("sqlite:///second.db", "Comparación", 23, "make"),
        (f"sqlite:///{stores[0]}", "Comparación", 42, "python"),
        (f"sqlite:///{stores[0]}", "Otro experimento", 31, "python"),
    ]:
        process = run_cli(cli_project, "--tracking-uri", uri, "--experiment-name", name,
                          "--seed", str(seed), runner=runner)
        assert process.returncode == 0, process.stderr
    first, second = [MlflowClient(tracking_uri=f"sqlite:///{store}") for store in stores]
    first_experiment = first.get_experiment_by_name("Comparación")
    second_experiment = second.get_experiment_by_name("Comparación")
    first_runs = first.search_runs([first_experiment.experiment_id])
    second_runs = second.search_runs([second_experiment.experiment_id])
    assert sorted(r.data.params["seed"] for r in first_runs) == ["17", "42"]
    assert [r.data.params["seed"] for r in second_runs] == ["23"]
    assert not {r.info.run_id for r in first_runs} & {r.info.run_id for r in second_runs}
    other = first.get_experiment_by_name("Otro experimento")
    assert [r.data.params["seed"] for r in first.search_runs([other.experiment_id])] == ["31"]
    assert second.get_experiment_by_name("Otro experimento") is None
    for client, run in [(first, first_runs[0]), (second, second_runs[0])]:
        assert Path(client.download_artifacts(run.info.run_id, "provenance/metadata.json")).is_file()
    assert not (cli_project / ".mlflow").exists()
    assert not (cli_project / "second.db").exists()
    assert not (cli_project.parent / "mlruns").exists()


@pytest.mark.parametrize("selection", ["experiment", "uri"])
def test_cli_can_override_each_tracking_option_independently(cli_project, loan_files, selection):
    from mlflow import MlflowClient

    copy_cli_inputs(cli_project, *loan_files[:2])
    name = "Solo experimento" if selection == "experiment" else "riesgo-impago-logistica"
    store = (cli_project / ".mlflow" / "mlflow.db") if selection == "experiment" else (
        cli_project.parent / "only-uri.db"
    )
    arguments = ["--experiment-name", name] if selection == "experiment" else [
        "--tracking-uri", f"sqlite:///{store}",
    ]
    process = run_cli(cli_project, *arguments)
    assert process.returncode == 0, process.stderr
    client = MlflowClient(tracking_uri=f"sqlite:///{store}")
    experiment = client.get_experiment_by_name(name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
    assert Path(client.download_artifacts(runs[0].info.run_id, "evaluation/metrics.json")).is_file()


@pytest.mark.parametrize("problem", ["invalid_uri", "inaccessible"])
def test_cli_tracking_destination_errors_do_not_report_success(cli_project, problem):
    blocked = cli_project.parent / "blocked"
    blocked.write_text("not a directory")
    uri = "unsupported-tracking://destination" if problem == "invalid_uri" else (
        f"sqlite:///{blocked / 'results.db'}"
    )
    process = run_cli(cli_project, "--tracking-uri", uri)
    assert process.returncode == 1
    assert "unsupported-tracking" in process.stderr if problem == "invalid_uri" else (
        "FileExistsError" in process.stderr or "NotADirectoryError" in process.stderr
    )
    assert "FileNotFoundError" not in process.stderr  # Tracking fails before CSV loading.
    assert "Corrida MLflow:" not in process.stdout
    assert "Modelo MLflow:" not in process.stdout
    assert not (cli_project / ".mlflow").exists()
    assert not (cli_project / "reports").exists()
    assert blocked.read_text() == "not a directory"


def test_cli_records_source_fingerprints_and_actual_counts(cli_project, loan_files):
    from hashlib import sha256
    from mlflow import MlflowClient

    copy_cli_inputs(cli_project, *loan_files[:2])
    features_path, status_path = (cli_project / "data" / name for name in ["X.csv", "target.csv"])
    features = pd.read_csv(features_path)
    pd.concat([features, features.iloc[[0, 2]]]).to_csv(features_path, index=False)
    statuses = pd.read_csv(status_path)
    pd.concat([statuses, pd.DataFrame({"y": [0, 1]})]).to_csv(status_path, index=False)
    originals = (features_path.read_bytes(), status_path.read_bytes())
    # Construct the client only after the executable initializes its local store.
    process = run_cli(cli_project)
    assert process.returncode == 0, process.stderr
    client = MlflowClient(tracking_uri=f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    run_id = next(line.removeprefix("Corrida MLflow: ") for line in process.stdout.splitlines()
                  if line.startswith("Corrida MLflow: "))
    run = client.get_run(run_id)
    assert run.data.tags["data.features_sha256"] == sha256(originals[0]).hexdigest()
    assert run.data.tags["data.payment_status_sha256"] == sha256(originals[1]).hexdigest()
    counts = {
        "source_rows": 62, "sampled_rows": 62, "duplicate_rows_removed": 1,
        "contradictory_rows_removed": 2, "retained_rows": 59,
        "training_rows": 47, "test_rows": 12,
    }
    assert {k: int(run.data.tags[f"data.{k}"]) for k in counts} == counts
    assert (features_path.read_bytes(), status_path.read_bytes()) == originals
    document = json.loads(Path(client.download_artifacts(run_id, "provenance/metadata.json")).read_text())
    assert document["data"] == {
        **counts, "features_sha256": sha256(originals[0]).hexdigest(),
        "payment_status_sha256": sha256(originals[1]).hexdigest(),
    }
    hashes = []
    for changed in [None, "features", "status"]:
        if changed == "features":
            frame = pd.read_csv(features_path)
            frame.loc[0, "funded_amnt"] += 1
            frame.to_csv(features_path, index=False)
        elif changed == "status":
            frame = pd.read_csv(status_path)
            frame.loc[1, "y"] = 0
            frame.to_csv(status_path, index=False)
        process = run_cli(cli_project)
        assert process.returncode == 0, process.stderr
        run_id = next(line.removeprefix("Corrida MLflow: ") for line in process.stdout.splitlines()
                      if line.startswith("Corrida MLflow: "))
        tags = client.get_run(run_id).data.tags
        hashes.append((tags["data.features_sha256"], tags["data.payment_status_sha256"]))
    assert hashes[0] == (sha256(originals[0]).hexdigest(), sha256(originals[1]).hexdigest())
    assert hashes[1][0] != hashes[0][0] and hashes[1][1] == hashes[0][1]
    assert hashes[2][0] == hashes[1][0] and hashes[2][1] != hashes[1][1]


@pytest.mark.parametrize("state", ["absent", "clean", "dirty", "staged", "untracked", "parent", "no_binary"])
def test_cli_records_versions_and_accurate_git_state(cli_project, loan_files, state, monkeypatch):
    from importlib.metadata import version
    from platform import python_version
    from mlflow import MlflowClient

    copy_cli_inputs(cli_project, *loan_files[:2])
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    commit = None
    if state not in ["absent", "no_binary"]:
        repo = cli_project.parent if state == "parent" else cli_project

        def git(*args):
            return subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", f"core.hooksPath={os.devnull}", "-C", str(repo), *args],
                check=True, capture_output=True, text=True,
            ).stdout.strip()

        git("init")
        (repo / ".gitignore").write_text(".mlflow/\nreports/\ndata/\n__pycache__/\n")
        git("add", ".")
        git("commit", "-m", "synthetic baseline")
        commit = git("rev-parse", "HEAD")
        if state in ["dirty", "staged"]:
            with (cli_project / "train.py").open("a") as script:
                script.write("\n# synthetic local change\n")
            if state == "staged":
                git("add", "train.py")
        if state == "untracked":
            (cli_project / "new-note.txt").write_text("uncommitted")
    if state == "no_binary":
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("GIT_PYTHON_REFRESH", "quiet")

    process = run_cli(cli_project)
    assert process.returncode == 0, process.stderr
    client = MlflowClient(tracking_uri=f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    experiment = client.get_experiment_by_name("riesgo-impago-logistica")
    run = client.search_runs([experiment.experiment_id])[0]
    document = json.loads(Path(client.download_artifacts(run.info.run_id, "provenance/metadata.json")).read_text())
    expected_git = {
        "status": "unavailable", "commit": None, "dirty": None,
    } if state in ["absent", "parent", "no_binary"] else {
        "status": "available", "commit": commit, "dirty": state != "clean",
    }
    assert document["git"] == expected_git
    assert run.data.tags["git.status"] == expected_git["status"]
    assert run.data.tags["git.commit"] == (expected_git["commit"] or "unavailable")
    assert run.data.tags["git.dirty"] == (
        "unavailable" if expected_git["dirty"] is None else str(expected_git["dirty"]).lower()
    )
    assert document["software"]["python"] == python_version()
    for package in ["mlflow", "numpy", "pandas", "scikit-learn", "matplotlib", "cloudpickle", "scipy", "joblib", "threadpoolctl"]:
        assert document["software"][package] == version(package)
        assert run.data.tags[f"software.{package}"] == version(package)


def test_cli_persists_a_complete_tracking_run(cli_project, loan_files, training_config):
    from mlflow import MlflowClient
    import mlflow
    import mlflow.sklearn

    feature_path, status_path, _ = loan_files
    copy_cli_inputs(cli_project, feature_path, status_path)
    expected = run_training(training_config)

    process = run_cli(cli_project)

    assert process.returncode == 0, process.stderr
    assert "Corrida MLflow:" in process.stdout
    client = MlflowClient(tracking_uri=f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    experiment = client.get_experiment_by_name("riesgo-impago-logistica")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params == {
        "C": "1.0", "class_weight": "balanced", "max_iter": "1000",
        "seed": "42", "solver": "lbfgs", "sample_cap": "100000",
        "test_fraction": "0.2", "threshold": "0.5", "dummy_strategy": "prior",
    }
    assert run.data.metrics == {
        **{f"test_logistic_{k}": v for k, v in expected.primary_evaluation.metrics.items()},
        **{f"test_dummy_{k}": v for k, v in expected.dummy_evaluation.metrics.items()},
    }
    assert {item.path for item in client.list_artifacts(run.info.run_id, "evaluation")} == {
        "evaluation/evaluation_baseline.png", "evaluation/metrics.json",
    }
    metrics_path = client.download_artifacts(run.info.run_id, "evaluation/metrics.json")
    assert json.loads(Path(metrics_path).read_text()) == {
        "metrics": run.data.metrics, "threshold": 0.5,
    }
    figure_path = client.download_artifacts(run.info.run_id, "evaluation/evaluation_baseline.png")
    assert Path(figure_path).read_bytes() == (
        cli_project / "reports" / "figures" / "evaluation_baseline.png"
    ).read_bytes()
    assert f"Corrida MLflow: {run.info.run_id}" in process.stdout
    model_uri = cli_model_uri(process)
    mlflow.set_tracking_uri(client.tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    probe = loan_files[2].iloc[:4]
    np.testing.assert_array_equal(model.predict(probe), expected.model.predict(probe))
    np.testing.assert_allclose(model.predict_proba(probe), expected.model.predict_proba(probe))
    repeated = run_cli(cli_project)
    assert repeated.returncode == 0, repeated.stderr
    repeated_runs = client.search_runs([experiment.experiment_id])
    assert len(repeated_runs) == 2
    assert len({r.info.run_id for r in repeated_runs}) == 2
    assert all(r.data.metrics == run.data.metrics for r in repeated_runs)
    assert not (cli_project.parent / ".mlflow").exists()


@pytest.mark.parametrize("stage", [
    "training", "figure", "json", "model", "packaging", "finalization", "figure_and_finalization",
    "provenance", "provenance_tags", "hash_read",
])
def test_cli_incomplete_tracking_runs_fail_without_success(cli_project, loan_files, stage):
    from mlflow import MlflowClient

    if stage != "training":
        copy_cli_inputs(cli_project, *loan_files[:2])
    # Inject faults only at the external SDK boundary, not training internals.
    process = subprocess.run(
        [sys.executable, "-c", """
import sys, runpy, shutil
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mlflow, mlflow.sklearn
stage = sys.argv[2]
original = mlflow.log_artifact
def log_artifact(path, *args, **kwargs):
    if ((stage.startswith('figure') and path.endswith('.png'))
            or (stage == 'json' and path.endswith('metrics.json'))
            or (stage == 'provenance' and path.endswith('metadata.json'))):
        raise RuntimeError('injected ' + stage + ' failure')
    return original(path, *args, **kwargs)
def log_model(*args, **kwargs):
    raise RuntimeError('injected model failure')
mlflow.log_artifact = log_artifact
if stage == 'provenance_tags':
    def set_tags(*args, **kwargs):
        raise RuntimeError('injected provenance_tags failure')
    mlflow.set_tags = set_tags
if stage == 'hash_read':
    original_open = Path.open
    def open_path(self, mode='r', *args, **kwargs):
        if mode == 'rb' and self.suffix == '.csv':
            raise PermissionError('injected hash_read failure')
        return original_open(self, mode, *args, **kwargs)
    Path.open = open_path
if stage == 'packaging':
    original_copyfile = shutil.copyfile
    def copyfile(source, destination, *args, **kwargs):
        if 'credit-risk-code-' in str(destination):
            raise PermissionError('injected packaging failure')
        return original_copyfile(source, destination, *args, **kwargs)
    shutil.copyfile = copyfile
if stage.endswith('finalization'):
    def set_terminated(*args, **kwargs):
        raise RuntimeError('injected finalization failure')
    mlflow.MlflowClient.set_terminated = set_terminated
if stage == 'model':
    mlflow.sklearn.log_model = log_model
sys.argv = [sys.argv[1] + '/train.py']
runpy.run_path(sys.argv[0], run_name='__main__')
""", str(cli_project), stage],
        cwd=cli_project.parent, capture_output=True, text=True, timeout=30,
    )

    assert process.returncode == 1
    assert "FileNotFoundError" in process.stderr if stage == "training" else (
        f"injected {stage} failure" in process.stderr
    )
    assert "Corrida MLflow:" not in process.stdout
    assert "Modelo MLflow:" not in process.stdout
    if stage == "figure_and_finalization":
        # Both failures must be visible, but cleanup cannot replace the primary error.
        assert "injected finalization failure" in process.stderr
        assert "RuntimeError: injected figure_and_finalization failure" == next(
            line for line in reversed(process.stderr.splitlines()) if line.startswith("RuntimeError:")
        )
    client = MlflowClient(tracking_uri=f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    experiment = client.get_experiment_by_name("riesgo-impago-logistica")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == ("RUNNING" if stage.endswith("finalization") else "FAILED")


def test_public_training_creates_no_tracking_state(training_config, local_tracking):
    run_training(training_config)

    assert not local_tracking.exists()


@pytest.mark.parametrize("destination", ["default", "selected"])
def test_registered_pipeline_loads_without_the_checkout_or_editable_package(
    cli_project, loan_files, training_config, tmp_path, destination,
):
    import sysconfig

    copy_cli_inputs(cli_project, *loan_files[:2])
    expected = run_training(training_config)
    tracking_uri = f"sqlite:///{cli_project}/.mlflow/mlflow.db" if destination == "default" else (
        f"sqlite:///{tmp_path / 'portable.db'}"
    )
    arguments = [] if destination == "default" else [
        "--tracking-uri", tracking_uri, "--experiment-name", "Portabilidad integrada",
    ]
    process = run_cli(cli_project, *arguments)
    assert process.returncode == 0, process.stderr
    model_uri = cli_model_uri(process)
    probe = loan_files[2].iloc[:4]
    probe_path = tmp_path / "synthetic_probe.json"
    probe.to_json(probe_path, orient="split")
    outside = tmp_path / "outside"
    outside.mkdir()
    # -I -S skips cwd, PYTHONPATH and .pth startup (including editable finders).
    loaded = subprocess.run(
        [sys.executable, "-I", "-S", "-c", """
import sys, importlib.util, json
from pathlib import Path
sys.path.append(sys.argv[1])
import mlflow.sklearn, pandas as pd
assert importlib.util.find_spec('credit_risk') is None, 'checkout leaked into process'
mlflow.set_tracking_uri(sys.argv[2])
model_path = mlflow.artifacts.download_artifacts(artifact_uri=sys.argv[3])
model = mlflow.sklearn.load_model(sys.argv[3])
import credit_risk.features
assert Path(credit_risk.features.__file__).is_relative_to(Path(model_path) / 'code')
probe = pd.read_json(sys.argv[4], orient='split')
Path('predictions.json').write_text(json.dumps({
    'predictions': model.predict(probe).tolist(),
    'probabilities': model.predict_proba(probe).tolist(),
}))
""", sysconfig.get_path("purelib"), tracking_uri, model_uri, str(probe_path)],
        cwd=outside, capture_output=True, text=True, timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    predictions = json.loads((outside / "predictions.json").read_text())
    np.testing.assert_array_equal(predictions["predictions"], expected.model.predict(probe))
    np.testing.assert_allclose(
        predictions["probabilities"], expected.model.predict_proba(probe), rtol=1e-12, atol=1e-12,
    )


def test_registered_model_describes_safe_raw_inputs_and_runtime_dependencies(cli_project, loan_files):
    from importlib.metadata import version
    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn

    copy_cli_inputs(cli_project, *loan_files[:2])
    # Files unrelated to Python source must not hitch a ride in code_paths.
    (cli_project / "credit_risk" / "private.csv").write_text("synthetic private sentinel")
    (cli_project / "credit_risk" / "credentials.json").write_text('{"secret": "sentinel"}')
    process = run_cli(cli_project)
    assert process.returncode == 0, process.stderr
    model_uri = cli_model_uri(process)
    mlflow.set_tracking_uri(f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    metadata = mlflow.models.Model.load(model_path)
    assert metadata.signature is not None
    example = metadata.load_input_example(model_path)
    assert isinstance(example, pd.DataFrame)
    assert example.shape == (1, 17)
    assert set(example.columns) == set(loan_files[2].columns)
    assert "y" not in example.columns
    assert example["emp_length"].tolist() == ["5 years"]
    assert not (loan_files[2]["funded_amnt"] == example.iloc[0]["funded_amnt"]).any()
    assert set(metadata.signature.inputs.input_names()) == set(example.columns)
    packaged_files = [p for p in (Path(model_path) / "code").rglob("*") if p.is_file()]
    assert packaged_files
    assert all(p.suffix == ".py" for p in packaged_files)
    assert not list(Path(model_path).rglob("*.csv"))
    assert not list(Path(model_path).rglob("credentials.json"))
    model = mlflow.sklearn.load_model(model_path)
    generic_model = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_equal(generic_model.predict(example), model.predict(example))
    assert set(model.predict(example)).issubset({0, 1})
    requirements_path = mlflow.pyfunc.get_model_dependencies(model_path, format="pip")
    requirements = set(Path(requirements_path).read_text().splitlines())
    for package in ["mlflow", "scikit-learn", "numpy", "pandas", "cloudpickle", "matplotlib"]:
        assert f"{package}=={version(package)}" in requirements
    assert all("==" in requirement for requirement in requirements)


@pytest.mark.parametrize("runner", ["python", "make"])
@pytest.mark.parametrize("option", ["-h", "--help"])
def test_cli_help_needs_no_data_and_displays_defaults(cli_project, option, runner):
    process = run_cli(cli_project, option, runner=runner)

    assert process.returncode == 0
    for option, default in [
        ("--C", "1.0"), ("--max-iter", "1000"),
        ("--seed", "42"), ("--class-weight", "balanced"),
        ("--feature-path", str(cli_project / "data" / "X.csv")),
        ("--payment-status-path", str(cli_project / "data" / "target.csv")),
        ("--figure-path", str(cli_project / "reports" / "figures" / "evaluation_baseline.png")),
        ("--experiment-name", "riesgo-impago-logistica"),
        ("--tracking-uri", f"sqlite:///{cli_project / '.mlflow' / 'mlflow.db'}"),
    ]:
        assert option in process.stdout
        # argparse can wrap long default paths across terminal lines.
        assert "".join(default.split()) in "".join(process.stdout.split())
    assert "Entrenamiento:" not in process.stdout
    assert not (cli_project / "reports").exists()
    assert not (cli_project / ".mlflow").exists()


@pytest.mark.parametrize("arguments, message", [
    (["--unknown"], "unrecognized arguments"),
    (["--max", "10"], "unrecognized arguments"),
    (["--seed"], "expected one argument"),
    (["--max-iter", "1.5"], "invalid int value"),
    (["--C", "not-a-number"], "invalid float value"),
    (["--C", "0"], "regularization"),
    (["--C", "nan"], "regularization"),
    (["--max-iter", "0"], "max_iterations"),
    (["--seed", "-1"], "seed"),
    (["--seed", "4294967296"], "seed"),
    (["--class-weight", "invalid"], "invalid choice"),
    (["--class-weight"], "expected one argument"),
    (["--class", "none"], "unrecognized arguments"),
    (["--feature-path"], "expected one argument"),
    (["--payment-status-path"], "expected one argument"),
    (["--figure-path"], "expected one argument"),
    (["--experiment-name"], "expected one argument"),
    (["--tracking-uri"], "expected one argument"),
])
def test_cli_rejects_arguments_before_reading_data(cli_project, arguments, message):
    process = run_cli(cli_project, *arguments)

    assert process.returncode == 2
    assert message in process.stderr
    assert "Traceback" not in process.stderr
    assert "FileNotFoundError" not in process.stderr
    assert not process.stdout
    assert not (cli_project / "reports").exists()
    assert not (cli_project / ".mlflow").exists()


@pytest.mark.parametrize("seed", [0, 4294967295])
def test_seed_limits_support_training(training_config, seed):
    run = run_training(replace(training_config, seed=seed))
    assert run.seed == seed
    assert run.training_rows == 48


@pytest.mark.parametrize("weight", ["none", "invalid", {}, 1, False, ["balanced"]])
def test_configuration_rejects_unsupported_payment_status_weighting(weight):
    with pytest.raises(ValueError, match="class_weight"):
        TrainingConfig(class_weight=weight)


def test_configuration_defaults_to_balanced_payment_status_weighting():
    assert TrainingConfig().class_weight == "balanced"


@pytest.fixture
def imbalanced_loan_files(loan_files):
    feature_path, status_path, features = loan_files
    rng = np.random.default_rng(92)
    # Overlapping outcomes and unequal frequencies expose the effect of weighting.
    for name, low, high in [
        ("funded_amnt", 2000, 15000), ("interest_rate", 5, 25),
        ("annual_income", 30000, 100000), ("dept_paym_income_ratio", 5, 40),
        ("used_credit_share", 0, 100),
    ]:
        features[name] = rng.uniform(low, high, len(features))
    features.to_csv(feature_path, index=False)
    pd.DataFrame({"y": rng.permutation([0] * 45 + [1] * 15)}).to_csv(status_path, index=False)
    return feature_path, status_path, features


def test_payment_status_weighting_changes_training_but_not_dummy_or_observations(
    imbalanced_loan_files, training_config,
):
    feature_path, status_path, _ = imbalanced_loan_files
    originals = (feature_path.read_bytes(), status_path.read_bytes())
    baseline = run_training(training_config)
    weighted = run_training(replace(training_config, class_weight="balanced"))
    unweighted = run_training(replace(training_config, class_weight=None))

    np.testing.assert_array_equal(
        baseline.primary_evaluation.probabilities, weighted.primary_evaluation.probabilities,
    )
    assert not np.allclose(
        weighted.primary_evaluation.probabilities, unweighted.primary_evaluation.probabilities,
    )
    assert weighted.primary_evaluation.metrics != unweighted.primary_evaluation.metrics
    np.testing.assert_array_equal(
        weighted.dummy_evaluation.probabilities, unweighted.dummy_evaluation.probabilities,
    )
    assert weighted.dummy_evaluation.metrics == unweighted.dummy_evaluation.metrics
    assert (feature_path.read_bytes(), status_path.read_bytes()) == originals


@pytest.mark.parametrize("argument, weight", [("balanced", "balanced"), ("none", None)])
def test_cli_payment_status_weighting_matches_repeated_public_training(
    cli_project, imbalanced_loan_files, training_config, argument, weight,
):
    feature_path, status_path, _ = imbalanced_loan_files
    copy_cli_inputs(cli_project, feature_path, status_path)
    originals = (feature_path.read_bytes(), status_path.read_bytes())
    copied_inputs = [cli_project / "data" / name for name in ["X.csv", "target.csv"]]
    copied_originals = tuple(path.read_bytes() for path in copied_inputs)
    config = replace(training_config, class_weight=weight, regularization=0.1, seed=17)
    expected = run_training(config)
    repeated = run_training(config)
    opposite = run_training(replace(config, class_weight=None if weight == "balanced" else "balanced"))
    assert [f"{v:.4f}" for v in expected.primary_evaluation.metrics.values()] != [
        f"{v:.4f}" for v in opposite.primary_evaluation.metrics.values()
    ], "Fixture must detect ignored weighting at CLI display precision"
    for first, second in [
        (expected.primary_evaluation, repeated.primary_evaluation),
        (expected.dummy_evaluation, repeated.dummy_evaluation),
    ]:
        assert first.metrics == second.metrics
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        np.testing.assert_array_equal(first.predictions, second.predictions)

    process = run_cli(
        cli_project, "--class-weight", argument, "--C", "0.1",
        "--max-iter", "1000", "--seed", "17",
    )

    assert process.returncode == 0, process.stderr
    for name, evaluation in [
        ("Logística", expected.primary_evaluation), ("Dummy", expected.dummy_evaluation),
    ]:
        line = next(line for line in process.stdout.splitlines() if line.startswith(name))
        assert line.split()[1:] == [f"{value:.4f}" for value in evaluation.metrics.values()]
    assert "Semilla: 17" in process.stdout
    assert (cli_project / "reports" / "figures" / "evaluation_baseline.png").stat().st_size > 0
    assert (feature_path.read_bytes(), status_path.read_bytes()) == originals
    assert tuple(path.read_bytes() for path in copied_inputs) == copied_originals
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{cli_project}/.mlflow/mlflow.db")
    experiment = client.get_experiment_by_name("riesgo-impago-logistica")
    persisted = client.search_runs([experiment.experiment_id])[0]
    assert persisted.data.params["class_weight"] == argument
    assert persisted.data.params["C"] == "0.1"
    assert persisted.data.params["seed"] == "17"
    assert persisted.data.metrics["test_logistic_AP"] == expected.primary_evaluation.metrics["AP"]


@pytest.mark.parametrize("arguments, overrides", [
    (["--C", "1e-8"], {"regularization": 1e-8}),
    (["--max-iter", "1"], {"max_iterations": 1}),
    (["--seed", "17"], {"seed": 17}),
    (["--C", "1e-8", "--max-iter", "1", "--seed", "17"],
     {"regularization": 1e-8, "max_iterations": 1, "seed": 17}),
])
def test_cli_arguments_produce_the_requested_evaluation(
    cli_project, loan_files, training_config, arguments, overrides,
):
    feature_path, status_path, features = loan_files
    # Independent predictors and overlapping outcomes make each option observable.
    rng = np.random.default_rng(71)
    for name, low, high in [
        ("funded_amnt", 2000, 15000), ("interest_rate", 5, 25),
        ("annual_income", 30000, 100000), ("dept_paym_income_ratio", 5, 40),
        ("used_credit_share", 0, 100),
    ]:
        features[name] = rng.uniform(low, high, len(features))
    features.to_csv(feature_path, index=False)
    outcomes = rng.integers(0, 2, len(features))
    pd.DataFrame({"y": outcomes}).to_csv(status_path, index=False)
    copy_cli_inputs(cli_project, feature_path, status_path)
    baseline = run_training(training_config)
    if overrides.get("max_iterations") == 1:
        with pytest.warns(UserWarning, match="converge"):
            expected = run_training(replace(training_config, **overrides))
    else:
        expected = run_training(replace(training_config, **overrides))
    expected_metrics = [f"{value:.4f}" for value in expected.primary_evaluation.metrics.values()]
    baseline_metrics = [f"{value:.4f}" for value in baseline.primary_evaluation.metrics.values()]
    assert expected_metrics != baseline_metrics, "Fixture must detect ignored arguments"

    process = run_cli(cli_project, *arguments)

    assert process.returncode == 0, process.stderr
    primary_line = next(line for line in process.stdout.splitlines() if line.startswith("Logística"))
    assert primary_line.split()[1:] == expected_metrics
    assert f"Semilla: {expected.seed}" in process.stdout
    assert "Entrenamiento: 48" in process.stdout
    assert "Prueba: 12" in process.stdout
    assert "Dummy" in process.stdout
    assert "Evidencia:" in process.stdout
    assert (cli_project / "reports" / "figures" / "evaluation_baseline.png").stat().st_size > 0
    if overrides.get("max_iterations") == 1:
        assert "converge" in process.stderr


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_does_not_relabel_data_errors_as_argument_errors(cli_project, loan_files, explicit):
    feature_path, status_path, _ = loan_files
    pd.DataFrame({"y": [0] * 60}).to_csv(status_path, index=False)
    copy_cli_inputs(cli_project, feature_path, status_path)

    arguments = [
        "--feature-path", str(cli_project / "data" / "X.csv"),
        "--payment-status-path", str(cli_project / "data" / "target.csv"),
    ] if explicit else []
    process = run_cli(cli_project, *arguments)

    assert process.returncode == 1
    assert "ambos estados de pago" in process.stderr
    assert "usage:" not in process.stderr
    assert not (cli_project / "reports").exists()


@pytest.mark.parametrize("runner", ["python", "make"])
@pytest.mark.parametrize("relative", [False, True])
def test_cli_accepts_explicit_paths_with_spaces_from_invoking_directory(
    cli_project, loan_files, relative, runner,
):
    feature_path, status_path, _ = loan_files
    inputs = cli_project.parent / "mis datos"
    inputs.mkdir()
    features = inputs / "características.csv"
    status = inputs / "estado de pago.csv"
    shutil.copy2(feature_path, features)
    shutil.copy2(status_path, status)
    figure = cli_project.parent / "mis resultados" / "evaluación.png"
    paths = [features, status, figure]
    values = [str(path.relative_to(cli_project.parent) if relative else path) for path in paths]
    originals = (features.read_bytes(), status.read_bytes())

    process = run_cli(
        cli_project, "--feature-path", values[0], "--payment-status-path", values[1],
        "--figure-path", values[2], "--C", "0.1", "--seed", "17", runner=runner,
    )

    assert process.returncode == 0, process.stderr
    assert "Semilla: 17" in process.stdout
    assert "Entrenamiento: 48" in process.stdout
    assert "Logística" in process.stdout and "Dummy" in process.stdout
    assert f"Evidencia: {values[2]}" in process.stdout
    assert figure.stat().st_size > 0
    assert not (cli_project / "reports").exists()
    assert (features.read_bytes(), status.read_bytes()) == originals


def test_cli_validates_configuration_before_reading_explicit_missing_inputs(cli_project):
    process = run_cli(
        cli_project, "--C", "0", "--feature-path", "missing.csv",
        "--payment-status-path", "missing-status.csv", "--figure-path", "new.png",
    )

    assert process.returncode == 2
    assert "regularization" in process.stderr
    assert "FileNotFoundError" not in process.stderr
    assert "Traceback" not in process.stderr
    assert not (cli_project.parent / "new.png").exists()


@pytest.mark.parametrize("source", ["features", "status"])
@pytest.mark.parametrize("alias", ["direct", "relative", "symlink", "hardlink"])
def test_cli_rejects_figure_collisions_without_modifying_inputs(
    cli_project, loan_files, source, alias,
):
    feature_path, status_path, _ = loan_files
    originals = (feature_path.read_bytes(), status_path.read_bytes())
    destination = feature_path if source == "features" else status_path
    if alias == "relative":
        destination = Path(".") / destination.name
    elif alias in ["symlink", "hardlink"]:
        link = cli_project.parent / "figure.png"
        if alias == "symlink":
            link.symlink_to(destination)
        else:
            os.link(destination, link)
        destination = link

    process = run_cli(
        cli_project, "--feature-path", str(feature_path),
        "--payment-status-path", str(status_path), "--figure-path", str(destination),
    )

    assert process.returncode == 2
    assert "no puede sobrescribir" in process.stderr
    assert "Traceback" not in process.stderr
    assert not (cli_project / "reports").exists()
    assert (feature_path.read_bytes(), status_path.read_bytes()) == originals


def test_training_returns_usable_model_and_primary_evaluation(loan_files, training_config):
    _, _, features = loan_files
    result = run_training(training_config)

    assert result.training_rows == 48
    assert result.test_rows == 12
    assert result.seed == 42
    assert result.primary_evaluation.accuracy == 1.0
    assert result.primary_evaluation.probabilities.shape == (12,)
    assert result.primary_evaluation.predictions.shape == (12,)
    np.testing.assert_array_equal(
        result.model.predict(features.iloc[:4]), [0, 1, 0, 1],
    )


def test_training_configuration_is_immutable():
    config = TrainingConfig()
    with pytest.raises(FrozenInstanceError):
        setattr(config, "seed", 7)


@pytest.mark.parametrize("overrides", [
    {"regularization": 0}, {"regularization": -1},
    {"regularization": float("nan")}, {"regularization": float("inf")},
    {"regularization": float("-inf")}, {"regularization": True},
    {"max_iterations": 0}, {"max_iterations": -1},
    {"max_iterations": 1.5}, {"max_iterations": True},
    {"seed": -1}, {"seed": 4294967296}, {"seed": 1.5}, {"seed": True},
])
def test_configuration_rejects_invalid_training_arguments(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)


def test_executable_renders_a_training_run(capsys, training_config, monkeypatch):
    from train import main

    monkeypatch.setattr(sys, "argv", ["host-program", "--unrelated-option"])
    main(training_config)

    output = capsys.readouterr().out
    assert "Semilla: 42" in output
    assert "Entrenamiento: 48" in output
    assert "Prueba: 12" in output
    assert "Logística" in output


def test_training_and_returned_model_accept_source_employment_lengths(loan_files, training_config):
    feature_path, _, features = loan_files
    raw_lengths = ["< 1 year", "1 year", *[f"{year} years" for year in range(2, 10)],
                   "10+ years", None]
    features["emp_length"] = raw_lengths * 5
    features.to_csv(feature_path, index=False)
    run = run_training(training_config)

    raw = pd.concat([features.iloc[[0]]] * 12, ignore_index=True)
    raw["emp_length"] = [f"  {length}  " if length else None for length in raw_lengths]
    numeric = raw.copy()
    numeric["emp_length"] = [0.5, 1, *range(2, 10), 10, np.nan]
    before = raw.copy(deep=True)

    np.testing.assert_allclose(run.model.predict_proba(raw), run.model.predict_proba(numeric))
    pd.testing.assert_frame_equal(raw, before)
    assert np.isfinite(run.primary_evaluation.probabilities).all()


@pytest.mark.parametrize("runner", ["python", "make"])
def test_default_executable_works_from_another_directory(loan_files, tmp_path, runner):
    # Run an isolated copy of the deliverable with small default-path inputs,
    # never the repository's large raw CSVs.
    feature_path, status_path, _ = loan_files
    project = tmp_path / "checkout"
    project.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "credit_risk", project / "credit_risk")
    shutil.copy2(source / "train.py", project / "train.py")
    (project / "data").mkdir()
    shutil.copy2(feature_path, project / "data" / "X.csv")
    shutil.copy2(status_path, project / "data" / "target.csv")

    command = [sys.executable, str(project / "train.py")]
    if runner == "make":
        shutil.copy2(source / "Makefile", project / "Makefile")
        command = ["make", "-f", str(project / "Makefile"), "train", f"PYTHON={sys.executable}"]
    process = subprocess.run(
        command, cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=30,
    )

    assert "Entrenamiento: 48" in process.stdout
    assert "Prueba: 12" in process.stdout
    assert "Logística" in process.stdout
    assert (project / "reports" / "figures" / "evaluation_baseline.png").stat().st_size > 0


@pytest.mark.parametrize("operation", [run_training, load_historical_loan_sample])
def test_loading_rejects_missing_predictors_with_actionable_error(loan_files, operation, training_config):
    feature_path, _, features = loan_files
    features.drop(columns="funded_amnt").to_csv(feature_path, index=False)

    with pytest.raises(ValueError, match="columnas.*funded_amnt"):
        operation(training_config)


@pytest.mark.parametrize("problem, message", [
    ("row_count", "filas.*60.*59"),
    ("extra_column", "única columna.*y"),
    ("wrong_name", "única columna.*y"),
    ("invalid_value", "estado de pago.*0.*1"),
    ("missing_value", "estado de pago.*0.*1"),
    ("one_outcome", "ambos estados de pago"),
])
@pytest.mark.parametrize("operation", [run_training, load_historical_loan_sample])
def test_loading_validates_payment_status_contract(loan_files, problem, message, operation, training_config):
    _, status_path, _ = loan_files
    status = pd.read_csv(status_path)
    if problem == "row_count":
        status = status.iloc[:-1]
    elif problem == "extra_column":
        status["extra"] = 0
    elif problem == "wrong_name":
        status = status.rename(columns={"y": "status"})
    elif problem == "invalid_value":
        status.loc[0, "y"] = 2
    elif problem == "missing_value":
        status.loc[0, "y"] = np.nan
    else:
        status["y"] = 0
    status.to_csv(status_path, index=False)

    with pytest.raises(ValueError, match=message):
        operation(training_config)


def test_exploration_and_training_use_a_reproducible_stratified_sample(loan_files, training_config):
    _, status_path, features = loan_files
    # A deliberately imbalanced source: 40 préstamos pagados, 20 impagados.
    pd.DataFrame({"y": [0] * 40 + [1] * 20}).to_csv(status_path, index=False)
    config = replace(training_config, sample_cap=30, seed=17)
    first = load_historical_loan_sample(config)
    second = load_historical_loan_sample(config)

    assert len(first.features) == 30
    assert first.data_quality.source_rows == 60
    assert first.data_quality.sampled_rows == 30
    assert first.data_quality.retained_rows == 30
    assert first.payment_status.value_counts().to_dict() == {0: 20, 1: 10}
    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_series_equal(first.payment_status, second.payment_status)
    pd.testing.assert_frame_equal(first.features, features.loc[first.features.index])
    original_status = pd.read_csv(status_path)["y"]
    pd.testing.assert_series_equal(
        first.payment_status, original_status.loc[first.features.index],
    )
    other = load_historical_loan_sample(replace(training_config, sample_cap=30, seed=18))
    assert not first.features.index.equals(other.features.index)

    run = run_training(config)
    repeated = run_training(config)
    assert (run.training_rows, run.test_rows) == (24, 6)
    assert run.seed == 17
    for first_evaluation, repeated_evaluation in [
        (run.primary_evaluation, repeated.primary_evaluation),
        (run.dummy_evaluation, repeated.dummy_evaluation),
    ]:
        assert first_evaluation.metrics == repeated_evaluation.metrics
        np.testing.assert_array_equal(
            first_evaluation.probabilities, repeated_evaluation.probabilities,
        )
        np.testing.assert_array_equal(
            first_evaluation.predictions, repeated_evaluation.predictions,
        )
    np.testing.assert_array_equal(
        run.model.predict_proba(features), repeated.model.predict_proba(features),
    )


@pytest.mark.parametrize("cap, expected_rows", [(100_000, 60), (60, 60), (59, 59)])
def test_sample_handles_sources_at_below_or_just_above_cap(loan_files, cap, expected_rows, training_config):
    _, _, features = loan_files
    sample = load_historical_loan_sample(replace(training_config, sample_cap=cap))

    assert len(sample.features) == expected_rows
    assert set(sample.payment_status) == {0, 1}
    if expected_rows == 60:
        pd.testing.assert_frame_equal(sample.features, features)


@pytest.mark.parametrize("cap", [0, -1, 1, 2.5, True])
def test_invalid_sample_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="sample_cap.*entero.*2"):
        TrainingConfig(sample_cap=cap)


@pytest.mark.parametrize("operation", [run_training, load_historical_loan_sample])
@pytest.mark.parametrize("filename", ["X.csv", "target.csv"])
def test_empty_csv_has_an_actionable_loading_error(loan_files, operation, filename, training_config):
    feature_path, _, _ = loan_files
    (feature_path.parent / filename).write_text("")

    with pytest.raises(ValueError, match=f"CSV.*{filename}.*vacío"):
        operation(training_config)


def test_sample_audits_all_source_features_before_training_split(loan_files, training_config):
    feature_path, status_path, features = loan_files
    features["source_note"] = "observado"
    features.loc[[0, 2], "source_note"] = np.nan
    # One exact copy; three contradictory records (including a repeated 0);
    # one distinct record sharing every model predictor with record 4.
    source = pd.concat([features, features.iloc[[0, 2, 2, 4]]], ignore_index=True)
    source.loc[63, "source_note"] = "distinto"
    status = pd.concat([
        pd.read_csv(status_path)["y"], pd.Series([0, 1, 0, 0], name="y"),
    ], ignore_index=True)
    source.to_csv(feature_path, index=False)
    status.to_frame().to_csv(status_path, index=False)
    before = (feature_path.read_bytes(), status_path.read_bytes())
    config = training_config

    sample = load_historical_loan_sample(config)
    expected = source.drop(index=[2, 60, 61, 62])
    pd.testing.assert_frame_equal(sample.features, expected)
    pd.testing.assert_series_equal(sample.payment_status, status.loc[expected.index])
    quality = sample.data_quality
    assert quality.source_rows == 64
    assert quality.sampled_rows == 64
    assert quality.duplicate_rows_removed == 1
    assert quality.contradictory_rows_removed == 3
    assert quality.retained_rows == 60

    run = run_training(config)
    assert run.data_quality == quality
    assert (run.training_rows, run.test_rows) == (48, 12)
    assert (feature_path.read_bytes(), status_path.read_bytes()) == before


@pytest.mark.parametrize("problem", ["one_outcome_after_audit", "too_few_per_outcome"])
def test_training_explains_when_cleaned_sample_cannot_be_split(loan_files, problem, training_config):
    feature_path, status_path, features = loan_files
    if problem == "one_outcome_after_audit":
        # Every impagado has an identical pagado counterpart and must disappear.
        source = pd.concat([features, features.iloc[1::2]], ignore_index=True)
        status = np.concatenate([np.tile([0, 1], 30), np.zeros(30, dtype=int)])
    else:
        source = features.iloc[:3]
        status = [0, 1, 0]
    source.to_csv(feature_path, index=False)
    pd.DataFrame({"y": status}).to_csv(status_path, index=False)

    with pytest.raises(ValueError, match="división estratificada.*estados de pago"):
        run_training(training_config)


def test_executable_reports_sample_quality_and_split_counts(
    loan_files, capsys, training_config, local_tracking,
):
    from train import main

    feature_path, status_path, features = loan_files
    pd.concat([features, features.iloc[[0]]]).to_csv(feature_path, index=False)
    status = pd.read_csv(status_path)
    pd.concat([status, status.iloc[[0]]]).to_csv(status_path, index=False)

    main(training_config)

    output = capsys.readouterr().out
    assert "Registros fuente: 61" in output
    assert "Muestra antes de auditoría: 61" in output
    assert "Copias exactas eliminadas: 1" in output
    assert "Registros contradictorios eliminados: 0" in output
    assert "Muestra retenida: 60" in output
    assert "Entrenamiento: 48" in output
    assert "Prueba: 12" in output


def test_categorical_whitespace_blanks_and_unseen_values_are_usable(loan_files, training_config):
    feature_path, _, features = loan_files
    features["home_ownership_status"] = [" RENT ", "", None, "   "] * 15
    features.to_csv(feature_path, index=False)
    model = run_training(training_config).model
    raw = features.iloc[:4].copy()
    raw["grade"] = [" A1 ", "G5  ", "  A1", "G5"]
    normalized = raw.copy()
    normalized["grade"] = ["A1", "G5", "A1", "G5"]
    normalized["home_ownership_status"] = ["RENT", np.nan, np.nan, np.nan]
    np.testing.assert_allclose(model.predict_proba(raw), model.predict_proba(normalized))
    raw["home_ownership_status"] = "UNSEEN"
    assert np.isfinite(model.predict_proba(raw)).all()


@pytest.mark.parametrize("column, invalid", [
    ("funded_amnt", 0), ("interest_rate", -1), ("annual_income", 0),
    ("dept_paym_income_ratio", -1), ("total_credit_revolving_bal", -10),
    ("used_credit_share", np.inf), ("used_credit_share", -np.inf),
    ("tot_num_credit_lines", "not a number"),
    ("num_30+_delinq_in_2yrs", 0.5), ("num_inq_in_6mths", 1.1),
    ("num_open_credit_lines", 2.5), ("num_derogatory_pub_rec", 0.2),
    ("tot_num_credit_lines", 3.1), ("loan_term_months", 48),
    ("emp_length", "unknown"),
])
def test_invalid_source_values_are_equivalent_to_missing_values(loan_files, column, invalid, training_config):
    feature_path, _, features = loan_files
    # Include an invalid value during fit as well as at prediction time.
    features[column] = features[column].astype(object)
    features.loc[0, column] = invalid
    features.to_csv(feature_path, index=False)
    model = run_training(training_config).model
    raw = features.iloc[:2].copy()
    raw[column] = invalid
    missing = raw.copy()
    missing[column] = np.nan
    np.testing.assert_allclose(model.predict_proba(raw), model.predict_proba(missing))


def test_training_matches_notebook_log_preprocessing_and_logistic_defaults(loan_files, training_config):
    feature_path, _, features = loan_files
    features["annual_income"] = np.geomspace(1000, 1_000_000, 60)
    features["total_credit_revolving_bal"] = np.expm1(np.linspace(0, 10, 60))
    features.loc[[4, 7, 10], "annual_income"] = np.nan
    features.to_csv(feature_path, index=False)
    model = run_training(training_config).model
    probe = features.iloc[:4].copy()
    probe["annual_income"] = [2000, 50000, 200000, 1_000_000]
    probe["total_credit_revolving_bal"] = [0, 100, 50000, 1_000_000]
    # Independent reference: notebook cells 43, 52, 53, fit on this fixture's
    # stratified 80/20 seed-42 split with scikit-learn 1.9.1, not project helpers.
    expected = [0.014828716390, 0.986110138402, 0.012817437058, 0.984554626561]
    np.testing.assert_allclose(model.predict_proba(probe)[:, 1], expected, atol=1e-6)


def test_fit_statistics_do_not_use_holdout_and_inference_does_not_refit(loan_files, training_config):
    feature_path, _, features = loan_files
    features["annual_income"] = np.geomspace(1000, 1_000_000, 60)
    features["total_credit_revolving_bal"] = np.arange(60) ** 3
    features.loc[[3, 5, 7], "annual_income"] = np.nan
    features.to_csv(feature_path, index=False)
    config = training_config
    first = run_training(config).model
    probe = features.iloc[[3, 5, 7]].copy()
    expected = first.predict_proba(probe)
    # Independently recorded holdout identities for this fixture and seed 42.
    holdout = [12, 14, 19, 20, 28, 29, 31, 38, 40, 51, 55, 57]
    features.loc[holdout, "annual_income"] = 1e15
    features.loc[holdout, "total_credit_revolving_bal"] = 1e12
    features.loc[holdout, "grade"] = "UNSEEN"
    features.to_csv(feature_path, index=False)
    second = run_training(config).model
    np.testing.assert_allclose(second.predict_proba(probe), expected)
    second.predict_proba(features.loc[holdout])
    np.testing.assert_allclose(second.predict_proba(probe), expected)


def test_all_missing_preprocessing_branches_still_produce_a_usable_model(loan_files, training_config):
    feature_path, _, features = loan_files
    for column in features.columns:
        if column != "funded_amnt":
            features[column] = np.nan
    features.to_csv(feature_path, index=False)
    model = run_training(training_config).model
    assert np.isfinite(model.predict_proba(features.iloc[:2])).all()


def test_post_origination_fields_do_not_influence_training_or_prediction(loan_files, training_config):
    feature_path, _, features = loan_files
    features["paym_rec_for_tot_amnt_fund"] = np.arange(60)
    features["remaining_princ_for_tot_amnt_fund"] = np.arange(60) * 1000
    features.to_csv(feature_path, index=False)
    config = training_config
    first = run_training(config).model
    expected = first.predict_proba(features.iloc[:4])
    features["paym_rec_for_tot_amnt_fund"] = -1e12
    features["remaining_princ_for_tot_amnt_fund"] = 1e15
    features.to_csv(feature_path, index=False)
    second = run_training(config).model
    np.testing.assert_allclose(second.predict_proba(features.iloc[:4]), expected)
    predictors_only = features.drop(columns=[
        "paym_rec_for_tot_amnt_fund", "remaining_princ_for_tot_amnt_fund",
    ])
    np.testing.assert_allclose(second.predict_proba(predictors_only.iloc[:4]), expected)


@pytest.mark.parametrize("column, values", [
    ("grade", ["A1", "G5"]), ("loan_term_months", [36, 60]),
])
def test_subgrade_and_permitted_terms_remain_usable_predictors(loan_files, column, values, training_config):
    feature_path, _, features = loan_files
    features[column] = values * 30
    features.to_csv(feature_path, index=False)
    model = run_training(training_config).model
    probe = pd.concat([features.iloc[[0]]] * 2, ignore_index=True)
    probe[column] = values
    probabilities = model.predict_proba(probe)[:, 1]
    assert probabilities[1] > probabilities[0] + 1e-5
    numeric_text = probe.copy()
    for name in numeric_text.select_dtypes(include="number").columns:
        numeric_text[name] = numeric_text[name].astype(str)
    np.testing.assert_allclose(model.predict_proba(numeric_text), model.predict_proba(probe))


def test_training_compares_logistic_and_prior_dummy_on_the_same_holdout(training_config):
    run = run_training(training_config)
    assert run.primary_evaluation.metrics == {
        "AP": 1.0, "ROC_AUC": 1.0, "precision": 1.0,
        "recall": 1.0, "F1": 1.0, "accuracy": 1.0,
    }
    assert run.dummy_evaluation.metrics == pytest.approx({
        "AP": 0.5, "ROC_AUC": 0.5, "precision": 0.5,
        "recall": 1.0, "F1": 2 / 3, "accuracy": 0.5,
    })
    assert run.primary_evaluation.threshold == run.dummy_evaluation.threshold == 0.5
    for evaluation in [run.primary_evaluation, run.dummy_evaluation]:
        assert evaluation.probabilities.shape == (run.test_rows,)
        assert evaluation.predictions.shape == (run.test_rows,)
    np.testing.assert_array_equal(run.dummy_evaluation.probabilities, np.full(12, 0.5))
    np.testing.assert_array_equal(run.dummy_evaluation.predictions, np.ones(12))


def test_threshold_changes_decisions_not_ranking_and_zero_division_is_safe(training_config):
    normal = run_training(training_config)
    no_positive = run_training(replace(training_config, threshold=1.0))
    for before, after in [
        (normal.primary_evaluation, no_positive.primary_evaluation),
        (normal.dummy_evaluation, no_positive.dummy_evaluation),
    ]:
        np.testing.assert_array_equal(after.predictions, np.zeros(12))
        np.testing.assert_array_equal(after.probabilities, before.probabilities)
        assert after.metrics["AP"] == before.metrics["AP"]
        assert after.metrics["ROC_AUC"] == before.metrics["ROC_AUC"]
        assert after.metrics["precision"] == after.metrics["recall"] == after.metrics["F1"] == 0
        assert after.accuracy == 0.5


def test_training_creates_only_a_nonempty_evaluation_figure(loan_files):
    feature_path, status_path, _ = loan_files
    destination = feature_path.parent / "reports" / "figures" / "evaluation.png"
    run = run_training(TrainingConfig(
        feature_path=feature_path, payment_status_path=status_path,
        figure_path=destination,
    ))
    assert run.figure_path == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert set(destination.parent.iterdir()) == {destination}


def test_executable_prints_comparison_table_and_figure_location(loan_files, capsys, local_tracking):
    from train import main

    feature_path, status_path, _ = loan_files
    destination = feature_path.parent / "evaluation.png"
    main(TrainingConfig(
        feature_path=feature_path, payment_status_path=status_path,
        figure_path=destination,
    ))
    output = capsys.readouterr().out
    lines = [line.split() for line in output.splitlines()]
    assert ["Modelo", "AP", "ROC_AUC", "precision", "recall", "F1", "accuracy"] in lines
    assert ["Logística", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"] in lines
    assert ["Dummy", "0.5000", "0.5000", "0.5000", "1.0000", "0.6667", "0.5000"] in lines
    assert f"Evidencia: {destination}" in output


@pytest.mark.parametrize("filename", ["X.csv", "target.csv"])
@pytest.mark.parametrize("alias", [None, "symlink", "hardlink"])
def test_figure_destination_cannot_overwrite_raw_csv(loan_files, filename, alias):
    feature_path, status_path, _ = loan_files
    destination = feature_path.parent / filename
    if alias:
        source = destination
        destination = feature_path.parent / "alias.png"
        if alias == "symlink":
            destination.symlink_to(source)
        else:
            destination.hardlink_to(source)
    before = (feature_path.read_bytes(), status_path.read_bytes())
    with pytest.raises(ValueError, match="figura.*CSV"):
        run_training(TrainingConfig(
            feature_path=feature_path, payment_status_path=status_path,
            figure_path=destination,
        ))
    assert (feature_path.read_bytes(), status_path.read_bytes()) == before


def test_prior_dummy_exposes_imbalance_without_predicting_impagados(loan_files, training_config):
    _, status_path, _ = loan_files
    pd.DataFrame({"y": np.tile([0, 0, 0, 0, 1], 12)}).to_csv(status_path, index=False)
    run = run_training(training_config)
    # This seed/split holds out 10 pagados and 2 impagados, training on 38/10.
    assert run.dummy_evaluation.metrics == pytest.approx({
        "AP": 1 / 6, "ROC_AUC": 0.5, "precision": 0.0,
        "recall": 0.0, "F1": 0.0, "accuracy": 5 / 6,
    })
    np.testing.assert_allclose(run.dummy_evaluation.probabilities, np.full(12, 10 / 48))
    np.testing.assert_array_equal(run.dummy_evaluation.predictions, np.zeros(12))


def test_exploration_notebook_runs_sequentially_on_temporary_loans(loan_files, tmp_path):
    feature_path, status_path, _ = loan_files
    notebook_path = (Path(__file__).resolve().parents[1] / "notebooks"
                     / "1.0-eaml-historical-loan-exploration.ipynb")
    notebook = json.loads(notebook_path.read_text())
    # Edit the documented configuration cell, as a notebook user would.
    for cell in notebook["cells"]:
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = (
                "from pathlib import Path\n"
                "config = TrainingConfig("
                f"feature_path=Path({str(feature_path)!r}), "
                f"payment_status_path=Path({str(status_path)!r}), sample_cap=40)"
            )
    notebook_path = tmp_path / "exploration.ipynb"
    notebook_path.write_text(json.dumps(notebook))
    original_inputs = (feature_path.read_bytes(), status_path.read_bytes())
    completed = subprocess.run(
        [sys.executable, "-c", """
import json, sys
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
for cell in json.load(open(sys.argv[1]))['cells']:
    if cell['cell_type'] == 'code':
        result = shell.run_cell(''.join(cell['source']), store_history=False)
        if result.error_before_exec or result.error_in_exec:
            sys.exit(1)
""", str(notebook_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=90,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Préstamos históricos retenidos: 40" in completed.stdout
    assert "Registros fuente: 60" in completed.stdout
    assert "Semilla: 42" in completed.stdout
    assert (feature_path.read_bytes(), status_path.read_bytes()) == original_inputs
    assert not list(tmp_path.rglob("*.png"))
