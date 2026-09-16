#!/usr/bin/env python3
"""Upload a file to Google Drive using the toolkit's Google OAuth client
(same client_secret as youtube_upload.py — project ai-video-publish).

One-time setup:
  1. Enable the Drive API on the project:
     https://console.cloud.google.com/apis/library/drive.googleapis.com?project=ai-video-publish
  2. python3 tools/gdrive_upload.py --auth   (browser consent, drive.file scope)

Usage:
  python3 tools/gdrive_upload.py --file video.mp4 [--name "Title.mp4"] [--folder <folderId>]

Scope is drive.file: the token can only see files it created — it cannot
read the rest of the Drive.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN = Path(__file__).parent.parent / "_internal" / ".youtube" / "token_drive.json"


def get_creds():
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN.write_text(creds.to_json())
    return creds if creds and creds.valid else None


def auth():
    secrets = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE") or str(
        Path.home() / ".config/youtube/client_secret.json"
    )
    flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
    creds = flow.run_local_server(port=8765, open_browser=False)
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    TOKEN.write_text(creds.to_json())
    print(f"Token saved: {TOKEN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--name")
    ap.add_argument("--folder", help="Drive folder ID to upload into (shared drives OK)")
    ap.add_argument("--mkdir", help="Create a folder with this name (optionally inside --folder) and print its ID")
    ap.add_argument("--list-shared-drives", action="store_true")
    args = ap.parse_args()

    if args.auth:
        auth()
        return

    creds = get_creds()
    if not creds:
        sys.exit("No valid token — run with --auth first")
    svc = build("drive", "v3", credentials=creds)

    if args.list_shared_drives:
        drives = svc.drives().list(pageSize=50).execute().get("drives", [])
        print(json.dumps([{"id": d["id"], "name": d["name"]} for d in drives], indent=2))
        return

    if args.mkdir:
        meta = {"name": args.mkdir, "mimeType": "application/vnd.google-apps.folder"}
        if args.folder:
            meta["parents"] = [args.folder]
        f = svc.files().create(body=meta, fields="id,name,webViewLink",
                               supportsAllDrives=True).execute()
        print(json.dumps(f, indent=2))
        return

    if not args.file:
        ap.error("--file required")
    meta = {"name": args.name or Path(args.file).name}
    if args.folder:
        meta["parents"] = [args.folder]
    media = MediaFileUpload(args.file, resumable=True)
    f = (
        svc.files()
        .create(body=meta, media_body=media, fields="id,name,webViewLink",
                supportsAllDrives=True)
        .execute()
    )
    print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()
