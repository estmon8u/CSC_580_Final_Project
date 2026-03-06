import json
import zipfile
from pathlib import Path

from tiny_dreamer_highway.evaluation import (
    copy_artifact_files,
    create_bundle_archive,
    export_submission_bundle,
    write_bundle_manifest,
)


def make_artifact_files(tmp_path: Path) -> dict[str, Path]:
    files = {
        "metrics_plot": tmp_path / "metrics.png",
        "comparison_grid": tmp_path / "comparison.png",
        "comparison_video": tmp_path / "comparison.gif",
    }
    for name, path in files.items():
        path.write_text(name, encoding="utf-8")
    return files


def test_copy_artifact_files_copies_named_artifacts(tmp_path: Path) -> None:
    artifacts = make_artifact_files(tmp_path)

    copied = copy_artifact_files(artifacts, tmp_path / "bundle")

    assert set(copied.keys()) == set(artifacts.keys())
    assert all(path.exists() for path in copied.values())


def test_write_bundle_manifest_records_files_and_metadata(tmp_path: Path) -> None:
    copied_files = copy_artifact_files(make_artifact_files(tmp_path), tmp_path / "bundle")

    manifest = write_bundle_manifest(
        tmp_path / "bundle" / "manifest.json",
        bundle_name="submission_bundle",
        copied_files=copied_files,
        metadata={"step": 1, "course": "CSC 580 AI 2"},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["bundle_name"] == "submission_bundle"
    assert payload["metadata"]["step"] == 1
    assert payload["files"]["comparison_video"] == "comparison.gif"


def test_create_bundle_archive_writes_zip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "artifact.txt").write_text("artifact", encoding="utf-8")

    archive = create_bundle_archive(bundle_dir, tmp_path / "bundle.zip")

    assert archive.exists()
    with zipfile.ZipFile(archive) as handle:
        assert "artifact.txt" in handle.namelist()


def test_export_submission_bundle_writes_bundle_manifest_and_archive(tmp_path: Path) -> None:
    artifacts = make_artifact_files(tmp_path)

    outputs = export_submission_bundle(
        artifacts,
        tmp_path / "exports",
        bundle_name="report_bundle",
        metadata={"seed": 7},
    )

    assert outputs["bundle_dir"].exists()
    assert outputs["manifest"].exists()
    assert outputs["archive"].exists()
    assert outputs["metrics_plot"].name == "metrics.png"