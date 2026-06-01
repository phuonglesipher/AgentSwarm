from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_tool_dir = str(Path(__file__).parent)
if _tool_dir not in sys.path:
    sys.path.insert(0, _tool_dir)

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


def _error(action: str, message: str, **extra: Any) -> tuple[str, dict[str, object]]:
    return f"drive-download error ({action}): {message}", {"error": message, "action": action, **extra}


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def drive_download(
        url_or_id: str,
        action: str = "download",
        format_filter: str = "",
        dest_folder: str = "",
        service_account_key: str = "",
        max_file_size_mb: int = 500,
    ) -> tuple[str, dict[str, object]]:
        """Download files from Google Drive.

        Accepts Google Drive share links, direct links, folder links, or
        raw file/folder IDs. For folders, enumerates contents and downloads
        each file (with optional format filtering). Public links work
        without credentials; private files require a service account key.

        Args:
            url_or_id: Google Drive URL or file/folder ID. Supports formats:
                - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
                - https://drive.google.com/drive/folders/FOLDER_ID
                - https://drive.google.com/open?id=FILE_ID
                - Raw file or folder ID string
            action: Operation to perform. One of:
                - download: Download file(s) to dest_folder.
                - list: List folder contents without downloading (folders only).
            format_filter: Comma-separated file extensions to include
                (e.g. ".pdf,.docx,.txt"). Empty string means all formats.
                Applied before download to avoid unnecessary transfers.
            dest_folder: Destination directory for downloads. Empty string
                uses artifact_root/drive-downloads. Absolute paths accepted.
            service_account_key: Path to a Google service account JSON key
                file for private file access. Empty string tries the
                GOOGLE_SERVICE_ACCOUNT_KEY environment variable, then
                falls back to public-only mode via gdown.
            max_file_size_mb: Maximum file size in MB to download (default
                500). Files exceeding this are skipped with a warning.
        """
        from drive_helpers import (
            download_file_authenticated,
            download_file_public,
            download_folder_authenticated,
            download_folder_public,
            get_drive_service,
            list_folder_contents,
            parse_drive_url,
            parse_format_filter,
        )

        # --- Resolve destination folder ---
        if dest_folder:
            dest = Path(dest_folder)
        else:
            dest = context.artifact_root / "drive-downloads"
        dest.mkdir(parents=True, exist_ok=True)

        # --- Parse URL ---
        try:
            resource_type, resource_id = parse_drive_url(url_or_id)
        except ValueError as e:
            return _error("parse", str(e), url=url_or_id)

        # --- Parse format filter ---
        extensions = parse_format_filter(format_filter)

        # --- Resolve credentials ---
        drive_service = get_drive_service(service_account_key)

        # --- Action: list ---
        if action == "list":
            if resource_type != "folder":
                return _error("list", "The 'list' action requires a folder URL or ID.")
            try:
                items = list_folder_contents(resource_id, drive_service, extensions)
            except Exception as e:
                return _error("list", str(e))
            return (
                f"Found {len(items)} file(s) in folder.",
                {"action": "list", "folder_id": resource_id, "files": items},
            )

        # --- Action: download ---
        if action == "download":
            if resource_type == "file":
                try:
                    if drive_service:
                        result = download_file_authenticated(
                            resource_id, dest, drive_service, extensions, max_file_size_mb,
                        )
                    else:
                        result = download_file_public(
                            resource_id, dest, extensions, max_file_size_mb,
                        )
                except ImportError as e:
                    return _error("download", str(e))
                except Exception as e:
                    return _error("download", str(e), file_id=resource_id)

                if result.get("skipped"):
                    return (
                        f"File skipped: {result['reason']}",
                        {"action": "download", "skipped": True, **result},
                    )
                return (
                    f"Downloaded: {result['filename']} ({result['size_mb']:.1f} MB) → {dest}",
                    {"action": "download", "dest_folder": str(dest), "files": [result]},
                )

            # Folder download
            try:
                if drive_service:
                    results = download_folder_authenticated(
                        resource_id, dest, drive_service, extensions, max_file_size_mb,
                    )
                else:
                    results = download_folder_public(
                        resource_id, dest, extensions, max_file_size_mb,
                    )
            except ImportError as e:
                return _error("download", str(e))
            except Exception as e:
                return _error("download", str(e), folder_id=resource_id)

            downloaded = [r for r in results if not r.get("skipped")]
            skipped = [r for r in results if r.get("skipped")]
            return (
                f"Downloaded {len(downloaded)} file(s), skipped {len(skipped)} → {dest}",
                {
                    "action": "download",
                    "folder_id": resource_id,
                    "downloaded": downloaded,
                    "skipped": skipped,
                    "dest_folder": str(dest),
                },
            )

        return _error(action, f"Unknown action: {action}. Supported: download, list.")

    return drive_download
