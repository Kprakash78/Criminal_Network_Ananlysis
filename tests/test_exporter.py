"""
Tests for M6_feature/exporter.py — Court Evidence ZIP Exporter
PS 26152 — AI-Powered Criminal Network Analysis System
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from M6_feature.exporter import export_case_zip, _sha256, _collect_files


class TestExportCaseZip:
    def test_creates_zip_file(self, tmp_path):
        """export_case_zip should create a ZIP file."""
        zip_path = export_case_zip("demo_case", str(tmp_path))
        assert Path(zip_path).exists()
        assert zip_path.endswith(".zip")

    def test_zip_contains_manifest(self, tmp_path):
        """ZIP must contain manifest.json."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names

    def test_zip_contains_signature(self, tmp_path):
        """ZIP must contain manifest.sig."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.sig" in names

    def test_zip_contains_public_key(self, tmp_path):
        """ZIP must contain public_key.pem."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "public_key.pem" in names

    def test_zip_contains_verify_md(self, tmp_path):
        """ZIP must contain VERIFY.md."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "VERIFY.md" in names

    def test_manifest_has_correct_hashes(self, tmp_path):
        """Manifest SHA-256 hashes must match actual file content in the ZIP."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))

            for entry in manifest["files"]:
                file_data = zf.read(entry["path"])
                computed_hash = hashlib.sha256(file_data).hexdigest()
                assert computed_hash == entry["sha256"], \
                    f"Hash mismatch for {entry['path']}: " \
                    f"expected {entry['sha256']}, got {computed_hash}"

    def test_manifest_has_case_id(self, tmp_path):
        """Manifest should include the case_id."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["case_id"] == "demo_case"

    def test_manifest_has_timestamp(self, tmp_path):
        """Manifest should include an exported_at timestamp."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert "exported_at" in manifest
            assert len(manifest["exported_at"]) > 0

    def test_zip_includes_raw_data(self, tmp_path):
        """ZIP should include raw data files."""
        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Should have at least some FIR files
            fir_files = [n for n in names if n.startswith("raw_data/firs/")]
            assert len(fir_files) > 0, "No FIR files found in ZIP"

    def test_zip_includes_parsed_outputs(self, tmp_path):
        """ZIP should include parsed outputs if available."""
        # Generate demo outputs first
        m1_output = REPO_ROOT / "demo_dataset" / "M1_output"
        if not m1_output.exists():
            pytest.skip("Demo M1 output not available")

        zip_path = export_case_zip("demo_case", str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            parsed = [n for n in names if n.startswith("parsed_outputs/")]
            assert len(parsed) > 0, "No parsed output files found in ZIP"

    def test_encrypted_zip(self, tmp_path):
        """Encrypted ZIP should be created when passphrase provided."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography library not available")

        zip_path = export_case_zip("demo_case", str(tmp_path),
                                   encrypt_passphrase="test_password_123")
        assert Path(zip_path).exists()


class TestCollectFiles:
    def test_returns_list_of_tuples(self):
        """_collect_files should return (archive_path, fs_path) tuples."""
        files = _collect_files("demo_case")
        assert isinstance(files, list)
        if files:
            archive_path, fs_path = files[0]
            assert isinstance(archive_path, str)
            assert isinstance(fs_path, Path)

    def test_archive_paths_are_relative(self):
        """Archive paths should not start with /."""
        files = _collect_files("demo_case")
        for archive_path, _ in files:
            assert not archive_path.startswith("/"), \
                f"Archive path should be relative: {archive_path}"


class TestSha256:
    def test_known_hash(self):
        """SHA-256 of known input should match expected output."""
        data = b"test data for hashing"
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(data) == expected

    def test_empty_input(self):
        """SHA-256 of empty bytes should be the known empty hash."""
        empty_hash = hashlib.sha256(b"").hexdigest()
        assert _sha256(b"") == empty_hash
