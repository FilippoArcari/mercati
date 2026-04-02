"""
upload_to_drive.py  — ThermodynamicAgent Colab-ready
=====================================================
Carica l'intero progetto 'mercati' su Google Drive,
escludendo file inutili (__pycache__, outputs/, .pyc, log...).

Setup (una-tantum):
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import os
import sys
import mimetypes
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ── Configurazione ────────────────────────────────────────────────────────────

DRIVE_FOLDER_NAME = "ThermodynamicAgent_Colab"
CREDENTIALS_FILE  = "credentials.json"
TOKEN_FILE        = "token.json"
SCOPES            = ["https://www.googleapis.com/auth/drive"]

# Singoli file nella root del progetto
ROOT_FILES = [
    "main.py",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "QUICK_REFERENCE.txt",
    "PROJECT_STRUCTURE.md",
    "VDW_INTEGRATION_GUIDE.md",
    "INTEGRATION_SUMMARY.md",
    "CONFIG_ORCHESTRATION.md",
    "CONFIG_RESPONSIBILITY_MAP.md",
    "TRAINING_CONFIG_COMPARISON.md",
    "todo",
]

# Cartelle intere da caricare (ricorsive)
FOLDERS_TO_UPLOAD = [
    "modelli",      # tutto il codice dei modelli
    "config",       # tutti i .yaml
    "checkpoints",  # pesi salvati (.pth, .npz)
    "results",      # grafici e csv (train + test)
]

# Estensioni/cartelle da escludere sempre
EXCLUDE_EXTENSIONS = {".pyc", ".log"}
EXCLUDE_DIRS       = {"__pycache__", "outputs"}  # outputs = solo log, inutile su Colab

# ── Autenticazione ────────────────────────────────────────────────────────────

def authenticate():
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_FILE).exists():
                print(f"[ERRORE] '{CREDENTIALS_FILE}' non trovato.")
                print("  → Scaricalo da Google Cloud Console (OAuth 2.0 Desktop App)")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print("[OK] Token salvato.")

    return build("drive", "v3", credentials=creds)

# ── Utility Drive ─────────────────────────────────────────────────────────────

def get_or_create_folder(service, name: str, parent_id: str = None) -> str:
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id)").execute().get("files", [])
    if results:
        fid = results[0]["id"]
        print(f"  [EXISTS] 📁 {name}")
        return fid
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    fid = service.files().create(body=meta, fields="id").execute()["id"]
    print(f"  [NEW]    📁 {name}")
    return fid


def upload_file(service, local_path: Path, parent_id: str) -> str:
    mime, _ = mimetypes.guess_type(str(local_path))
    mime = mime or "application/octet-stream"

    query = f"name='{local_path.name}' and '{parent_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)

    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        print(f"    [↑ AGG]  {local_path.name}")
        return existing[0]["id"]
    else:
        meta = {"name": local_path.name, "parents": [parent_id]}
        fid = service.files().create(body=meta, media_body=media, fields="id").execute()["id"]
        print(f"    [↑ NEW]  {local_path.name}")
        return fid


def upload_folder_recursive(service, local_folder: Path, parent_id: str):
    if not local_folder.exists():
        print(f"  [SKIP] Cartella non trovata: {local_folder}")
        return

    drive_id = get_or_create_folder(service, local_folder.name, parent_id)

    for item in sorted(local_folder.iterdir()):
        if item.name in EXCLUDE_DIRS:
            continue
        if item.is_file():
            if item.suffix in EXCLUDE_EXTENSIONS:
                continue
            upload_file(service, item, drive_id)
        elif item.is_dir():
            if item.name in EXCLUDE_DIRS:
                continue
            upload_folder_recursive(service, item, drive_id)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ThermodynamicAgent → Google Drive")
    print("=" * 60)

    service = authenticate()
    root_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)

    # File singoli nella root
    print("\n── Root files ──")
    for fname in ROOT_FILES:
        p = Path(fname)
        if p.exists():
            upload_file(service, p, root_id)
        else:
            print(f"    [SKIP] {fname}")

    # Cartelle
    for folder in FOLDERS_TO_UPLOAD:
        print(f"\n── {folder}/ ──")
        upload_folder_recursive(service, Path(folder), root_id)

    print()
    print("=" * 60)
    print(f"[DONE] https://drive.google.com/drive/folders/{root_id}")
    print()
    print("── In Colab ─────────────────────────────────────────────")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")
    print(f"import sys")
    print(f"sys.path.insert(0, '/content/drive/MyDrive/{DRIVE_FOLDER_NAME}')")
    print("=" * 60)


if __name__ == "__main__":
    main()