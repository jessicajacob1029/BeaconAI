"""Download the official DDXPlus release into data/raw/ddxplus/.

Download links and MD5 checksums come from the Figshare API for the article id in
config/base.yaml, so no file URLs are hard-coded. If anything fails (no network, API change,
bad checksum), the command prints manual download instructions and exits with code 1.

Run: uv run python -m beaconai.data.download
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from beaconai.config import PROJECT_ROOT, DataSource, load_config

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
DATASET_SUBDIR = "ddxplus"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    md5: str | None
    size: int | None


def raw_dir_from_config(raw: Path) -> Path:
    raw = raw if raw.is_absolute() else PROJECT_ROOT / raw
    return raw / DATASET_SUBDIR


def patients_csv_name(zip_name: str) -> str:
    """release_train_patients.zip -> release_train_patients.csv"""
    return Path(zip_name).stem + ".csv"


def expected_outputs(files: list[str]) -> list[str]:
    """Files that must exist in the raw dir once download + extraction are done."""
    return [patients_csv_name(f) if f.endswith(".zip") else f for f in files]


def fetch_file_index(article_id: int, timeout: float) -> dict[str, RemoteFile]:
    """Ask the Figshare API which files the article has, with URLs and checksums."""
    try:
        with urlopen(FIGSHARE_API.format(article_id=article_id), timeout=timeout) as resp:
            payload = json.load(resp)
    except Exception as exc:
        raise DownloadError(f"Could not reach the Figshare API: {exc}") from exc
    files = payload.get("files")
    if not isinstance(files, list):
        raise DownloadError("Figshare API response has no 'files' list")
    index = {}
    for f in files:
        try:
            index[f["name"]] = RemoteFile(
                name=f["name"],
                url=f["download_url"],
                md5=f.get("computed_md5") or None,
                size=f.get("size"),
            )
        except (KeyError, TypeError) as exc:
            raise DownloadError(f"Unexpected Figshare file entry: {f!r}") from exc
    return index


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, timeout: float) -> None:
    """Stream a URL to `dest`, writing to a .part file first so failures leave no partial file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, CHUNK)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Download failed for {dest.name}: {exc}") from exc
    tmp.replace(dest)


def extract_patients_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the single CSV inside a patients zip to `<stem>.csv`.

    The archive member name is never used as a path, so a malicious name cannot write
    outside `dest_dir`.
    """
    out = dest_dir / patients_csv_name(zip_path.name)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if len(members) != 1:
                raise DownloadError(f"{zip_path.name}: expected 1 file, found {len(members)}")
            tmp = out.with_suffix(".csv.part")
            with zf.open(members[0]) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            tmp.replace(out)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"{zip_path.name} is not a valid zip file") from exc
    return out


def ensure_raw_files(
    raw_dir: Path,
    source: DataSource,
    fetch_index: Callable[[int, float], dict[str, RemoteFile]] = fetch_file_index,
    download: Callable[[str, Path, float], None] = download_url,
) -> list[Path]:
    """Make sure every expected raw file exists in `raw_dir`. Returns their paths.

    Files already present are not downloaded again. Network calls are injectable for tests.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [raw_dir / name for name in expected_outputs(source.files)]

    # Zips placed here by hand (manual download) are extracted without any network call.
    for name in source.files:
        local_zip = raw_dir / name
        if name.endswith(".zip") and local_zip.is_file():
            if not (raw_dir / patients_csv_name(name)).is_file():
                extract_patients_zip(local_zip, raw_dir)
            local_zip.unlink()

    if all(p.is_file() for p in targets):
        return targets

    index = fetch_index(source.figshare_article_id, source.timeout_seconds)
    missing_remote = [f for f in source.files if f not in index]
    if missing_remote:
        raise DownloadError(f"Files not listed on Figshare: {', '.join(missing_remote)}")

    for name in source.files:
        final = raw_dir / (patients_csv_name(name) if name.endswith(".zip") else name)
        if final.is_file():
            continue
        remote = index[name]
        dest = raw_dir / name
        if not dest.is_file():
            print(f"Downloading {name} ...", flush=True)
            download(remote.url, dest, source.timeout_seconds)
        if remote.md5 and md5_of(dest) != remote.md5:
            dest.unlink()
            raise DownloadError(f"Checksum mismatch for {name} (file removed, try again)")
        if name.endswith(".zip"):
            extract_patients_zip(dest, raw_dir)
            dest.unlink()

    still_missing = [p.name for p in targets if not p.is_file()]
    if still_missing:
        raise DownloadError(f"Still missing after download: {', '.join(still_missing)}")
    return targets


def manual_instructions(raw_dir: Path, source: DataSource) -> str:
    files = "\n".join(f"    - {f}" for f in source.files)
    return f"""
Automatic download of DDXPlus failed. To download it by hand:

  1. Open https://doi.org/{source.doi} (the official DDXPlus release on Figshare).
  2. Download these files:
{files}
  3. Put them in: {raw_dir}
     (zip files can stay zipped; they are extracted on the next run)
  4. Run `make data` again.
"""


def main() -> int:
    cfg = load_config()
    raw_dir = raw_dir_from_config(cfg.data.paths.raw)
    try:
        paths = ensure_raw_files(raw_dir, cfg.data.source)
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(manual_instructions(raw_dir, cfg.data.source), file=sys.stderr)
        return 1
    print("DDXPlus raw files ready:")
    for p in paths:
        print(f"  {p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
