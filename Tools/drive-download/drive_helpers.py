"""Google Drive download helpers: URL parsing, public/authenticated download, filtering."""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

_FILE_PATTERNS = [
    re.compile(r"/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]
_FOLDER_PATTERN = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_RAW_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,}$")


def parse_drive_url(url_or_id: str) -> tuple[str, str]:
    """Parse a Google Drive URL or raw ID into (resource_type, resource_id).

    Returns:
        ("file", file_id) or ("folder", folder_id)

    Raises:
        ValueError: If the URL format is not recognized.
    """
    url_or_id = url_or_id.strip()
    if not url_or_id:
        raise ValueError("Empty URL or ID provided.")

    # Folder URL
    m = _FOLDER_PATTERN.search(url_or_id)
    if m:
        return ("folder", m.group(1))

    # File URL patterns
    for pattern in _FILE_PATTERNS:
        m = pattern.search(url_or_id)
        if m:
            return ("file", m.group(1))

    # Raw ID (no URL structure)
    if _RAW_ID_PATTERN.match(url_or_id):
        return ("file", url_or_id)

    raise ValueError(
        f"Unrecognized Google Drive URL or ID: {url_or_id!r}. "
        "Expected a share link, folder link, or raw file/folder ID."
    )


# ---------------------------------------------------------------------------
# Format filter
# ---------------------------------------------------------------------------


def parse_format_filter(format_filter: str) -> set[str]:
    """Parse ".pdf,.docx" into {".pdf", ".docx"} (lowercased, dot-prefixed).

    Empty string returns empty set (no filter = accept all).
    """
    if not format_filter or not format_filter.strip():
        return set()
    extensions: set[str] = set()
    for part in format_filter.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.add(ext)
    return extensions


def _matches_filter(filename: str, extensions: set[str]) -> bool:
    """Check if filename matches the extension filter. Empty filter = accept all."""
    if not extensions:
        return True
    return Path(filename).suffix.lower() in extensions


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

# Google Docs native mimetypes → export format
_EXPORT_MIMETYPES: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
}


def get_drive_service(service_account_key: str) -> Any | None:
    """Build an authenticated Google Drive API service, or None for public-only.

    Resolution order:
      1. Explicit ``service_account_key`` path
      2. ``GOOGLE_SERVICE_ACCOUNT_KEY`` env var
      3. Returns None (public-only mode via gdown)
    """
    key_path = service_account_key.strip() if service_account_key else ""
    if not key_path:
        key_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY", "").strip()
    if not key_path:
        return None

    resolved = Path(key_path)
    if not resolved.is_file():
        return None

    try:
        from google.oauth2 import service_account  # type: ignore[import-untyped]
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        creds = service_account.Credentials.from_service_account_file(
            str(resolved),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public downloads (gdown)
# ---------------------------------------------------------------------------


def _ensure_gdown() -> Any:
    try:
        import gdown  # type: ignore[import-untyped]

        return gdown
    except ImportError:
        raise ImportError(
            "The 'gdown' package is required. Install with: pip install gdown>=5.0"
        )


def download_file_public(
    file_id: str,
    dest: Path,
    extensions: set[str],
    max_size_mb: int,
) -> dict[str, Any]:
    """Download a single public file via gdown."""
    gdown = _ensure_gdown()

    url = f"https://drive.google.com/uc?id={file_id}"

    # Attempt HEAD request for size check
    try:
        import requests

        resp = requests.head(url, allow_redirects=True, timeout=10)
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            return {
                "skipped": True,
                "reason": f"File exceeds {max_size_mb} MB limit",
                "file_id": file_id,
            }
    except Exception:
        pass  # HEAD may fail for Drive; proceed with download

    output_path = str(dest) + "/"
    downloaded = gdown.download(id=file_id, output=output_path, quiet=True, fuzzy=False)

    if not downloaded:
        return {"skipped": True, "reason": "gdown returned no file (permission or not found)", "file_id": file_id}

    dl_path = Path(downloaded)
    if not dl_path.exists():
        return {"skipped": True, "reason": "Downloaded file not found on disk", "file_id": file_id}

    # Extension filter
    if not _matches_filter(dl_path.name, extensions):
        dl_path.unlink(missing_ok=True)
        return {
            "skipped": True,
            "reason": f"Extension '{dl_path.suffix}' not in filter {extensions}",
            "filename": dl_path.name,
            "file_id": file_id,
        }

    # Size check post-download
    size_mb = dl_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        dl_path.unlink(missing_ok=True)
        return {
            "skipped": True,
            "reason": f"File size {size_mb:.1f} MB exceeds {max_size_mb} MB limit",
            "filename": dl_path.name,
            "file_id": file_id,
        }

    return {
        "filename": dl_path.name,
        "path": str(dl_path),
        "size_mb": round(size_mb, 2),
        "file_id": file_id,
    }


def download_folder_public(
    folder_id: str,
    dest: Path,
    extensions: set[str],
    max_size_mb: int,
) -> list[dict[str, Any]]:
    """Download all matching files from a public folder via gdown."""
    gdown = _ensure_gdown()

    downloaded_paths = gdown.download_folder(
        id=folder_id, output=str(dest), quiet=True, remaining_ok=True
    )

    if not downloaded_paths:
        return []

    results: list[dict[str, Any]] = []
    for fpath in downloaded_paths:
        p = Path(fpath)
        if not p.exists():
            continue

        # Extension filter
        if not _matches_filter(p.name, extensions):
            p.unlink(missing_ok=True)
            results.append({
                "skipped": True,
                "reason": f"Extension '{p.suffix}' not in filter {extensions}",
                "filename": p.name,
            })
            continue

        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            p.unlink(missing_ok=True)
            results.append({
                "skipped": True,
                "reason": f"File size {size_mb:.1f} MB exceeds {max_size_mb} MB limit",
                "filename": p.name,
            })
            continue

        results.append({
            "filename": p.name,
            "path": str(p),
            "size_mb": round(size_mb, 2),
        })

    return results


# ---------------------------------------------------------------------------
# Authenticated downloads (Drive API)
# ---------------------------------------------------------------------------


def _get_file_metadata(service: Any, file_id: str) -> dict[str, str]:
    """Fetch file metadata from Drive API."""
    return (
        service.files()
        .get(fileId=file_id, fields="id,name,size,mimeType", supportsAllDrives=True)
        .execute()
    )


def _download_binary(service: Any, file_id: str, dest_path: Path) -> None:
    """Download a binary (non-native) file via get_media."""
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _export_native(service: Any, file_id: str, mime_type: str, dest_path: Path) -> None:
    """Export a Google Docs/Sheets/Slides file to a binary format."""
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]

    request = service.files().export_media(fileId=file_id, mimeType=mime_type)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def download_file_authenticated(
    file_id: str,
    dest: Path,
    service: Any,
    extensions: set[str],
    max_size_mb: int,
) -> dict[str, Any]:
    """Download a single file via the Drive API."""
    meta = _get_file_metadata(service, file_id)
    name: str = meta.get("name", file_id)
    mime_type: str = meta.get("mimeType", "")
    raw_size: str = meta.get("size", "0")

    # Determine if this is a native Google file that needs export
    export_info = _EXPORT_MIMETYPES.get(mime_type)
    if export_info:
        export_mime, export_ext = export_info
        # Adjust filename to exported extension
        stem = Path(name).stem
        name = f"{stem}{export_ext}"
    else:
        export_mime = ""

    # Extension filter
    if not _matches_filter(name, extensions):
        return {
            "skipped": True,
            "reason": f"Extension '{Path(name).suffix}' not in filter {extensions}",
            "filename": name,
            "file_id": file_id,
        }

    # Size check (native Google files have no size field)
    if raw_size and raw_size != "0":
        size_mb = int(raw_size) / (1024 * 1024)
        if size_mb > max_size_mb:
            return {
                "skipped": True,
                "reason": f"File size {size_mb:.1f} MB exceeds {max_size_mb} MB limit",
                "filename": name,
                "file_id": file_id,
            }

    dest_path = dest / name

    if export_mime:
        _export_native(service, file_id, export_mime, dest_path)
    else:
        _download_binary(service, file_id, dest_path)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    return {
        "filename": name,
        "path": str(dest_path),
        "size_mb": round(size_mb, 2),
        "file_id": file_id,
    }


def _list_folder_api(
    service: Any, folder_id: str
) -> list[dict[str, str]]:
    """Paginate through all files in a Drive folder."""
    items: list[dict[str, str]] = []
    page_token: str | None = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, size, mimeType)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_folder_authenticated(
    folder_id: str,
    dest: Path,
    service: Any,
    extensions: set[str],
    max_size_mb: int,
) -> list[dict[str, Any]]:
    """Enumerate and download files from a folder via the Drive API."""
    items = _list_folder_api(service, folder_id)
    results: list[dict[str, Any]] = []

    for item in items:
        fid = item["id"]
        try:
            result = download_file_authenticated(fid, dest, service, extensions, max_size_mb)
            results.append(result)
        except Exception as e:
            results.append({
                "skipped": True,
                "reason": f"Download error: {e}",
                "filename": item.get("name", fid),
                "file_id": fid,
            })

    return results


# ---------------------------------------------------------------------------
# Folder listing (metadata only)
# ---------------------------------------------------------------------------


def list_folder_contents(
    folder_id: str,
    service: Any | None,
    extensions: set[str],
) -> list[dict[str, str]]:
    """List files in a folder without downloading.

    Uses Drive API if service is available, otherwise gdown folder listing.
    Returns list of dicts with keys: name, id, mimeType, size.
    """
    if service:
        items = _list_folder_api(service, folder_id)
        if extensions:
            items = [
                item for item in items
                if _matches_filter(item.get("name", ""), extensions)
            ]
        return items

    # Fallback: gdown folder listing (limited metadata)
    gdown = _ensure_gdown()
    try:
        from gdown.download_folder import _parse_google_drive_folder_url  # type: ignore[import-untyped]
        # gdown doesn't expose a clean list API; download with remaining_ok and inspect
        # For listing only, we return what we can from the folder page
    except ImportError:
        pass

    # gdown doesn't have a dedicated list-only API, so we do a dry-run style approach
    # by using the Google Drive folder page parser if available
    return [{"name": "(listing unavailable without credentials)", "id": folder_id}]
