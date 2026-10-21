"""Downloader tests. Network calls are replaced by fakes; nothing leaves the machine."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from beaconai.config import DataSource
from beaconai.data import download as dl
from beaconai.data.download import (
    DownloadError,
    RemoteFile,
    ensure_raw_files,
    expected_outputs,
    extract_patients_zip,
    manual_instructions,
)

CSV_BYTES = b"AGE,SEX\n30,F\n"
JSON_BYTES = b"{}"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


ZIP_BYTES = _zip_bytes({"release_test_patients": CSV_BYTES})
CONTENT = {"release_evidences.json": JSON_BYTES, "release_test_patients.zip": ZIP_BYTES}

SOURCE = DataSource(
    figshare_article_id=123,
    doi="10.0000/fake.123",
    files=list(CONTENT),
    timeout_seconds=1,
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class FakeRemote:
    """Stands in for the Figshare API and file host."""

    def __init__(self, content: dict[str, bytes], md5s: dict[str, str] | None = None) -> None:
        self.content = content
        self.md5s = md5s or {k: _md5(v) for k, v in content.items()}
        self.index_calls = 0
        self.downloads: list[str] = []

    def fetch_index(self, article_id: int, timeout: float) -> dict[str, RemoteFile]:
        self.index_calls += 1
        return {
            name: RemoteFile(name, f"fake://{name}", self.md5s[name], len(data))
            for name, data in self.content.items()
        }

    def download(self, url: str, dest: Path, timeout: float) -> None:
        name = url.removeprefix("fake://")
        self.downloads.append(name)
        dest.write_bytes(self.content[name])


def test_expected_outputs_replaces_zips_with_csv() -> None:
    assert expected_outputs(SOURCE.files) == [
        "release_evidences.json",
        "release_test_patients.csv",
    ]


def test_downloads_verifies_and_extracts(tmp_path: Path) -> None:
    remote = FakeRemote(CONTENT)
    paths = ensure_raw_files(tmp_path, SOURCE, remote.fetch_index, remote.download)
    assert [p.name for p in paths] == expected_outputs(SOURCE.files)
    assert (tmp_path / "release_test_patients.csv").read_bytes() == CSV_BYTES
    assert not (tmp_path / "release_test_patients.zip").exists()  # zip removed after extract
    assert sorted(remote.downloads) == sorted(CONTENT)


def test_existing_files_skip_network(tmp_path: Path) -> None:
    (tmp_path / "release_evidences.json").write_bytes(JSON_BYTES)
    (tmp_path / "release_test_patients.csv").write_bytes(CSV_BYTES)
    remote = FakeRemote(CONTENT)
    ensure_raw_files(tmp_path, SOURCE, remote.fetch_index, remote.download)
    assert remote.index_calls == 0
    assert remote.downloads == []


def test_manually_placed_zip_extracted_without_network(tmp_path: Path) -> None:
    (tmp_path / "release_evidences.json").write_bytes(JSON_BYTES)
    (tmp_path / "release_test_patients.zip").write_bytes(ZIP_BYTES)

    def no_network(*args: object) -> dict[str, RemoteFile]:
        raise AssertionError("network should not be used")

    ensure_raw_files(tmp_path, SOURCE, no_network, no_network)  # type: ignore[arg-type]
    assert (tmp_path / "release_test_patients.csv").read_bytes() == CSV_BYTES


def test_only_missing_files_downloaded(tmp_path: Path) -> None:
    (tmp_path / "release_evidences.json").write_bytes(JSON_BYTES)
    remote = FakeRemote(CONTENT)
    ensure_raw_files(tmp_path, SOURCE, remote.fetch_index, remote.download)
    assert remote.downloads == ["release_test_patients.zip"]


def test_checksum_mismatch_removes_file_and_raises(tmp_path: Path) -> None:
    remote = FakeRemote(CONTENT, md5s={k: "0" * 32 for k in CONTENT})
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        ensure_raw_files(tmp_path, SOURCE, remote.fetch_index, remote.download)
    assert not (tmp_path / "release_evidences.json").exists()


def test_file_missing_from_figshare_raises(tmp_path: Path) -> None:
    remote = FakeRemote({"release_evidences.json": JSON_BYTES})
    with pytest.raises(DownloadError, match="not listed"):
        ensure_raw_files(tmp_path, SOURCE, remote.fetch_index, remote.download)


def test_zip_with_several_members_rejected(tmp_path: Path) -> None:
    z = tmp_path / "release_test_patients.zip"
    z.write_bytes(_zip_bytes({"a": b"1", "b": b"2"}))
    with pytest.raises(DownloadError, match="expected 1 file"):
        extract_patients_zip(z, tmp_path)


def test_zip_member_name_cannot_escape_dest(tmp_path: Path) -> None:
    dest = tmp_path / "raw"
    dest.mkdir()
    z = dest / "release_test_patients.zip"
    z.write_bytes(_zip_bytes({"../../evil.csv": CSV_BYTES}))
    out = extract_patients_zip(z, dest)
    assert out == dest / "release_test_patients.csv"
    assert not (tmp_path / "evil.csv").exists()


def test_bad_zip_raises(tmp_path: Path) -> None:
    z = tmp_path / "release_test_patients.zip"
    z.write_bytes(b"not a zip")
    with pytest.raises(DownloadError, match="not a valid zip"):
        extract_patients_zip(z, tmp_path)


def test_manual_instructions_list_everything(tmp_path: Path) -> None:
    text = manual_instructions(tmp_path, SOURCE)
    assert "https://doi.org/10.0000/fake.123" in text
    assert str(tmp_path) in text
    for name in SOURCE.files:
        assert name in text


def test_main_prints_manual_instructions_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise DownloadError("no network")

    monkeypatch.setattr(dl, "raw_dir_from_config", lambda raw: tmp_path)
    monkeypatch.setattr(dl, "ensure_raw_files", failing)
    assert dl.main() == 1
    err = capsys.readouterr().err
    assert "no network" in err
    assert "Automatic download of DDXPlus failed" in err
    assert "https://doi.org/" in err
