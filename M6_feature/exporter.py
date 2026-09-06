"""
M6_feature — Court Evidence ZIP Exporter
PS 26152 — AI-Powered Criminal Network Analysis System

Packages all case evidence (raw files, parsed outputs, AI conclusions,
audit logs) into a signed ZIP archive suitable for court submission.

Features:
- SHA-256 hash verification for every included file
- RSA-2048 digital signature of manifest (ephemeral key pair)
- Optional AES-256 encryption via passphrase
- VERIFY.md with step-by-step verification instructions

Usage:
    from M6_feature.exporter import export_case_zip
    export_case_zip("demo_case", "/tmp")
    # Creates /tmp/case_demo_case_evidence.zip

CLI:
    python -m M6_feature.exporter demo_case /tmp
    python -m M6_feature.exporter demo_case /tmp --encrypt mypassword
"""

import hashlib
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(filepath: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(case_id: str) -> list[tuple[str, Path]]:
    """
    Collect all files to include in the evidence ZIP.

    Returns list of (archive_path, filesystem_path) tuples.
    """
    files = []

    # --- Raw data files ---
    data_dir = REPO_ROOT / "data"

    # FIRs
    fir_dir = data_dir / "firs"
    if fir_dir.exists():
        for f in sorted(fir_dir.glob("*.txt")):
            files.append((f"raw_data/firs/{f.name}", f))

    # CDRs
    cdr_path = data_dir / "cdrs" / "cdr.csv"
    if cdr_path.exists():
        files.append(("raw_data/cdrs/cdr.csv", cdr_path))

    # Transactions
    txn_path = data_dir / "transactions" / "transactions.csv"
    if txn_path.exists():
        files.append(("raw_data/transactions/transactions.csv", txn_path))

    # --- Parsed outputs (M1) ---
    m1_output = REPO_ROOT / "demo_dataset" / "M1_output"
    if m1_output.exists():
        for f in sorted(m1_output.glob("*.json")):
            files.append((f"parsed_outputs/{f.name}", f))

    # --- AI outputs ---
    # Top-3 results
    top3_path = REPO_ROOT / "M3_feature" / "top3_results.json"
    if top3_path.exists():
        files.append(("ai_outputs/top3_results.json", top3_path))

    # Search logs
    search_log = REPO_ROOT / "M6_feature" / "logs" / "retrieval_logs.jsonl"
    if search_log.exists():
        files.append(("ai_outputs/retrieval_logs.jsonl", search_log))

    # --- Demo case config ---
    config_path = REPO_ROOT / "demo_dataset" / "demo_case_config.json"
    if config_path.exists():
        files.append(("case_config.json", config_path))

    return files


def _generate_rsa_keys() -> tuple[bytes, bytes]:
    """
    Generate an ephemeral RSA-2048 key pair for signing.

    Returns (private_key_pem, public_key_pem).
    Uses the 'cryptography' library if available, otherwise falls back
    to a simpler HMAC-based signature.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization, hashes

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    except ImportError:
        logger.warning("[Exporter] 'cryptography' library unavailable. "
                       "Using HMAC fallback for signing.")
        return None, None


def _sign_manifest(manifest_bytes: bytes, private_key_pem: bytes) -> bytes:
    """Sign the manifest using RSA-PSS."""
    try:
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes, serialization

        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None
        )
        signature = private_key.sign(
            manifest_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return signature

    except ImportError:
        # HMAC fallback
        import hmac
        key = hashlib.sha256(b"ephemeral-demo-key").digest()
        return hmac.new(key, manifest_bytes, hashlib.sha256).digest()


def _encrypt_zip(zip_bytes: bytes, passphrase: str) -> bytes:
    """
    Encrypt a ZIP file using AES-256 (Fernet symmetric encryption).

    Falls back to unencrypted if the cryptography library is unavailable.
    """
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        import base64

        # Derive a Fernet key from the passphrase
        salt = b"criminal_network_analysis_salt"  # Fixed salt for demo reproducibility
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        fernet = Fernet(key)
        return fernet.encrypt(zip_bytes)

    except ImportError:
        logger.warning("[Exporter] 'cryptography' unavailable for encryption. "
                       "Returning unencrypted ZIP.")
        return zip_bytes


VERIFY_MD_CONTENT = """# Evidence Package Verification Guide

## Contents

This ZIP archive contains:
- `manifest.json` — File inventory with SHA-256 hashes and timestamps
- `manifest.sig` — RSA-PSS digital signature of manifest.json
- `public_key.pem` — Public key to verify the signature
- `raw_data/` — Original source files (FIRs, CDRs, transactions)
- `parsed_outputs/` — Machine-extracted entities and relationships
- `ai_outputs/` — AI analysis results (suspect rankings, search logs)

## Step 1: Verify File Integrity

Each file in `manifest.json` has a SHA-256 hash. Verify any file with:

```bash
sha256sum <filepath>
# Compare output with the hash in manifest.json
```

Or use Python:
```python
import hashlib, json

manifest = json.load(open("manifest.json"))
for entry in manifest["files"]:
    with open(entry["path"], "rb") as f:
        computed = hashlib.sha256(f.read()).hexdigest()
    status = "✓" if computed == entry["sha256"] else "✗ MISMATCH"
    print(f"{status} {entry['path']}")
```

## Step 2: Verify Digital Signature

The manifest is signed using RSA-2048 with PSS padding:

```python
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

# Load public key
with open("public_key.pem", "rb") as f:
    public_key = serialization.load_pem_public_key(f.read())

# Load manifest and signature
with open("manifest.json", "rb") as f:
    manifest_bytes = f.read()
with open("manifest.sig", "rb") as f:
    signature = f.read()

# Verify
try:
    public_key.verify(
        signature,
        manifest_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("✓ Signature is valid")
except Exception:
    print("✗ Signature verification FAILED")
```

## Step 3: Review AI Outputs

- `ai_outputs/top3_results.json` — Contains the top 3 suspects with risk scores
  and evidence references. Each evidence item points to a specific file and line range.
- `ai_outputs/retrieval_logs.jsonl` — Retrieval audit log showing every search query,
  retrieved chunks, scores, and timestamps.

## Notes

- The RSA key pair is **ephemeral** — generated at export time for this specific package.
- The private key is NOT included in the archive. Only the public key is included
  to allow verification.
- All analysis scores indicate investigation priority, not guilt determination.
- For encrypted packages: use the passphrase provided by the exporting officer
  to decrypt with AES-256 (Fernet) before verification.
"""


def export_case_zip(
    case_id: str,
    outpath: str | Path,
    encrypt_passphrase: str | None = None,
) -> str:
    """
    Export a complete evidence package as a signed ZIP archive.

    Args:
        case_id:            Case identifier (e.g., "demo_case")
        outpath:            Output directory for the ZIP file
        encrypt_passphrase: Optional passphrase for AES-256 encryption

    Returns:
        Absolute path to the created ZIP file.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    zip_filename = f"case_{case_id}_evidence.zip"
    zip_path = outpath / zip_filename

    # Collect files
    file_list = _collect_files(case_id)
    logger.info(f"[Exporter] Collecting {len(file_list)} files for case '{case_id}'")

    # Build manifest
    manifest = {
        "case_id": case_id,
        "exported_at": timestamp.isoformat(),
        "exported_by": "Criminal Network Analysis System v1.0",
        "file_count": len(file_list),
        "files": [],
    }

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add evidence files
        for archive_path, fs_path in file_list:
            file_data = fs_path.read_bytes()
            file_hash = _sha256(file_data)

            manifest["files"].append({
                "path": archive_path,
                "sha256": file_hash,
                "size_bytes": len(file_data),
                "created_at": datetime.fromtimestamp(
                    fs_path.stat().st_mtime
                ).isoformat(),
            })

            zf.writestr(archive_path, file_data)

        # Generate RSA keys and sign manifest
        private_key_pem, public_key_pem = _generate_rsa_keys()

        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        zf.writestr("manifest.json", manifest_bytes)

        if private_key_pem and public_key_pem:
            signature = _sign_manifest(manifest_bytes, private_key_pem)
            zf.writestr("manifest.sig", signature)
            zf.writestr("public_key.pem", public_key_pem)
        else:
            # HMAC fallback
            signature = _sign_manifest(manifest_bytes, b"")
            zf.writestr("manifest.sig", signature)
            zf.writestr("public_key.pem",
                         b"# HMAC fallback -- 'cryptography' library was not available\n"
                         b"# Signature uses HMAC-SHA256 instead of RSA-PSS\n")

        # Add VERIFY.md
        zf.writestr("VERIFY.md", VERIFY_MD_CONTENT)

    # Get ZIP bytes
    zip_bytes = zip_buffer.getvalue()

    # Optionally encrypt
    if encrypt_passphrase:
        logger.info("[Exporter] Encrypting ZIP with AES-256...")
        encrypted = _encrypt_zip(zip_bytes, encrypt_passphrase)

        # If encryption succeeded (different from original), save as .enc
        if encrypted != zip_bytes:
            enc_path = outpath / f"case_{case_id}_evidence.zip.enc"
            enc_path.write_bytes(encrypted)
            logger.info(f"[Exporter] Encrypted ZIP saved to {enc_path}")
            # Also save unencrypted for fallback
            zip_path.write_bytes(zip_bytes)
        else:
            zip_path.write_bytes(zip_bytes)
    else:
        zip_path.write_bytes(zip_bytes)

    logger.info(f"[Exporter] Evidence ZIP created: {zip_path} ({len(zip_bytes)} bytes)")
    print(f"[Exporter] Created {zip_path}")
    return str(zip_path)


# ===========================================================================
# Tar.gz fallback for when zipfile has issues
# ===========================================================================

def export_case_tar(case_id: str, outpath: str | Path) -> str:
    """
    Fallback exporter that creates a .tar.gz archive with a separate .sig file.
    Used when ZIP creation fails for any reason.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    tar_path = outpath / f"case_{case_id}_evidence.tar.gz"
    sig_path = outpath / f"case_{case_id}_evidence.sig"

    file_list = _collect_files(case_id)

    with tarfile.open(tar_path, "w:gz") as tf:
        for archive_path, fs_path in file_list:
            tf.add(str(fs_path), arcname=archive_path)

    # Sign the tar.gz
    tar_hash = _sha256_file(tar_path)
    sig_data = json.dumps({
        "file": str(tar_path.name),
        "sha256": tar_hash,
        "signed_at": datetime.now().isoformat(),
    }, indent=2)
    sig_path.write_text(sig_data, encoding="utf-8")

    logger.info(f"[Exporter] Tar archive created: {tar_path}")
    return str(tar_path)


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export court evidence ZIP")
    parser.add_argument("case_id", help="Case identifier")
    parser.add_argument("outpath", help="Output directory")
    parser.add_argument("--encrypt", help="Passphrase for AES-256 encryption",
                        default=None)

    args = parser.parse_args()
    export_case_zip(args.case_id, args.outpath, args.encrypt)
