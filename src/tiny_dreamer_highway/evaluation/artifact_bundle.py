"""Helpers for packaging report-ready evaluation artifacts.

Name: Esteban
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Mapping


def copy_artifact_files(
    artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    for name, source in artifacts.items():
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"artifact does not exist: {source_path}")
        destination = directory / source_path.name
        shutil.copy2(source_path, destination)
        copied[name] = destination
    return copied


def write_bundle_manifest(
    output_path: str | Path,
    *,
    bundle_name: str,
    copied_files: Mapping[str, Path],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_name": bundle_name,
        "files": {name: file_path.name for name, file_path in copied_files.items()},
        "metadata": dict(metadata or {}),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def create_bundle_archive(bundle_dir: str | Path, archive_path: str | Path) -> Path:
    directory = Path(bundle_dir)
    if not directory.exists():
        raise FileNotFoundError(f"bundle directory does not exist: {directory}")

    path = Path(archive_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(directory))
    return path


def export_submission_bundle(
    artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    bundle_name: str = "final_report_bundle",
    metadata: Mapping[str, object] | None = None,
    create_archive: bool = True,
) -> dict[str, Path]:
    bundle_dir = Path(output_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_files = copy_artifact_files(artifacts, bundle_dir)
    manifest_path = write_bundle_manifest(
        bundle_dir / "manifest.json",
        bundle_name=bundle_name,
        copied_files=copied_files,
        metadata=metadata,
    )

    outputs: dict[str, Path] = {
        "bundle_dir": bundle_dir,
        "manifest": manifest_path,
        **copied_files,
    }
    if create_archive:
        archive_path = create_bundle_archive(bundle_dir, bundle_dir.with_suffix(".zip"))
        outputs["archive"] = archive_path
    return outputs