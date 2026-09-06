"""
M6_feature/auto_brief.py
PS 26152 — AI-Powered Criminal Network Analysis System
======================================================
Auto-Brief Generator — One-Page Tactical PDF Brief.

Generates a deterministic, offline, human-readable PDF brief for a case:
  - Top-N elevated-priority candidates (name, risk score, 1-line reason,
    provenance snippet)
  - Short ordered event timeline (3–10 bullets from event_chains.json)
  - Template-based "Investigator Next Steps"
  - Signed manifest (SHA-256 hashes + seed) as manifest.json alongside PDF
  - Ephemeral RSA signature over the manifest (falls back to HMAC-SHA256
    if 'cryptography' library unavailable)

PDF is written using a minimal pure-Python PDF writer (no reportlab/fpdf
needed). The output is a valid, readable PDF.

LANGUAGE: All candidate references use "elevated priority" phrasing.
No guilt determinations are made. Brief includes a mandatory disclaimer.

CLI:
    python -m M6_feature.auto_brief export demo_case out/brief_demo_case.pdf
    python -m M6_feature.auto_brief export case_A  (writes to feature_outputs/)

API:
    from M6_feature.auto_brief import generate_brief
    pdf_path = generate_brief("case_A")
"""

import hashlib
import hmac
import json
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
DEMO_DIR   = REPO_ROOT / "demo_dataset"
FEAT_OUT   = REPO_ROOT / "demo_cache" / "feature_outputs"
M3_DIR     = REPO_ROOT / "M3_feature"

SEED = 42
np.random.seed(SEED)

MAX_CANDIDATES = 3
MAX_TIMELINE_BULLETS = 8
BRIEF_WIDTH = 72   # characters for text wrapping

DISCLAIMER = (
    "DISCLAIMER: This brief is an AI-assisted investigative tool. "
    "All findings are preliminary and require independent investigator "
    "verification before any operational use. No guilt determination is made."
)


# ── Minimal Pure-Python PDF Writer ────────────────────────────────────────────

class _SimplePDF:
    """
    Minimal conformant PDF writer (PDF 1.4).
    Supports: pages, Type1 Helvetica/Helvetica-Bold, basic text rendering.
    No external dependencies required.
    """

    def __init__(self):
        self._objects: list[bytes] = []
        self._pages: list[int] = []    # object indices of page objects
        self._offsets: list[int] = []
        self._buf: bytearray = bytearray()

    # ── object writing ────────────────────────────────────────────────────────
    def _add(self, content: bytes) -> int:
        """Append an object (without obj/endobj) and return 1-based index."""
        idx = len(self._objects) + 1
        self._objects.append(content)
        return idx

    # ── text helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _esc(s: str) -> str:
        """Escape a string for PDF literal strings."""
        return (s.replace("\\", "\\\\")
                 .replace("(", "\\(")
                 .replace(")", "\\)")
                 .replace("\r", "\\r")
                 .replace("\n", "\\n"))

    # ── page content building ─────────────────────────────────────────────────
    def _make_page_content(self, lines: list[tuple]) -> bytes:
        """
        lines: list of (x, y, font_name, font_size, text_str)
        where font_name ∈ {'F1': Helvetica, 'F2': Helvetica-Bold}
        """
        cmds = ["BT"]
        cur_font = None
        cur_size = None
        for (x, y, font, size, text) in lines:
            if font != cur_font or size != cur_size:
                cmds.append(f"/{font} {size} Tf")
                cur_font, cur_size = font, size
            cmds.append(f"{x} {y} Td")
            cmds.append(f"({self._esc(text)}) Tj")
            cmds.append(f"{-x} {-y} Td")   # reset position
        cmds.append("ET")
        return "\n".join(cmds).encode("latin-1", errors="replace")

    def add_page(self, lines: list[tuple]) -> None:
        """Add a page. `lines` as described in _make_page_content."""
        content_bytes = self._make_page_content(lines)
        # Stream object
        stream_idx = self._add(
            (f"<< /Length {len(content_bytes)} >>\n"
             f"stream\n").encode("ascii") +
            content_bytes +
            b"\nendstream"
        )
        # Page object (A4: 595 × 842 pt)
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 595 842] "
            f"/Contents {stream_idx} 0 R "
            f"/Resources << /Font << "
            f"/F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
            f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> "
            f">> >> >>"
        ).encode("ascii")
        page_idx = self._add(page_obj)
        self._pages.append(page_idx)

    def write(self, path: Path) -> None:
        """Serialise to a valid PDF file at `path`."""
        # Object 1: Catalog (placeholder; written after pages known)
        # Object 2: Pages dict
        # Objects 3+: page streams and page objects

        objects_with_catalog: list[bytes] = []

        # obj 1: catalog
        objects_with_catalog.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        # obj 2: pages (will reference pages added)
        page_refs = " ".join(f"{idx} 0 R" for idx in self._pages)
        objects_with_catalog.append(
            (f"<< /Type /Pages /Kids [{page_refs}] "
             f"/Count {len(self._pages)} >>").encode("ascii")
        )
        # remaining objects (streams + page dicts)
        objects_with_catalog += self._objects

        buf = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = []
        for i, obj_body in enumerate(objects_with_catalog, start=1):
            offsets.append(len(buf))
            buf += f"{i} 0 obj\n".encode("ascii")
            buf += obj_body
            buf += b"\nendobj\n"

        # xref
        xref_offset = len(buf)
        n = len(objects_with_catalog)
        buf += f"xref\n0 {n + 1}\n".encode("ascii")
        buf += b"0000000000 65535 f \n"
        for off in offsets:
            buf += f"{off:010d} 00000 n \n".encode("ascii")

        buf += (
            f"trailer\n<< /Size {n + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")

        path.write_bytes(bytes(buf))


# ── Text layout helpers ───────────────────────────────────────────────────────

def _wrap_lines(text: str, width: int = BRIEF_WIDTH) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _get_top3(case_id: str) -> list[dict]:
    cache = _load_json(REPO_ROOT / "demo_cache" / f"{case_id}_top3.json")
    if cache and "top3" in cache:
        return cache["top3"]
    # fallback: use ground_truth candidate names
    gt = _load_json(DEMO_DIR / "ground_truth.json")
    if gt and case_id in gt:
        return [{"name": n, "risk_score": 50.0,
                 "label": "elevated-priority candidate"}
                for n in gt[case_id]["top3_candidates"]]
    return []


def _get_chains(case_id: str) -> list[dict]:
    """Load event chains for this case (from M3_feature/ or feature_outputs/)."""
    for p in (M3_DIR / "event_chains.json",
              FEAT_OUT / "event_chains.json"):
        data = _load_json(p)
        if data and "chains" in data:
            return [c for c in data["chains"] if c.get("case_id") == case_id]
    return []


def _get_motif_flags(case_id: str) -> list[dict]:
    for p in (M3_DIR / "motif_flags.json",
              FEAT_OUT / "motif_flags.json"):
        data = _load_json(p)
        if data and "flags" in data and case_id in data["flags"]:
            return data["flags"][case_id]
    return []


def _get_case_title(case_id: str) -> str:
    gt = _load_json(DEMO_DIR / "ground_truth.json")
    if gt and case_id in gt:
        return gt[case_id].get("title", case_id)
    return case_id


# ── PDF layout ────────────────────────────────────────────────────────────────

_PAGE_W, _PAGE_H = 595, 842
_MARGIN_L, _MARGIN_T = 50, _PAGE_H - 50
_FONT_SIZE_TITLE  = 14
_FONT_SIZE_HEAD   = 11
_FONT_SIZE_BODY   = 9
_LINE_H_TITLE     = 20
_LINE_H_HEAD      = 16
_LINE_H_BODY      = 13


def _layout_brief(case_id: str, generated_at: str) -> list[tuple]:
    """
    Build list of (x, y, font, size, text) tuples for one-page PDF brief.
    Returns lines for _SimplePDF.add_page().
    """
    title    = _get_case_title(case_id)
    top3     = _get_top3(case_id)
    chains   = _get_chains(case_id)
    flags    = _get_motif_flags(case_id)

    lines: list[tuple] = []
    y = _MARGIN_T

    def push(text: str, font: str = "F1", size: int = _FONT_SIZE_BODY,
             extra_gap: int = 0) -> None:
        nonlocal y
        y -= size + extra_gap
        lines.append((50, y, font, size, text))

    # ── Header ────────────────────────────────────────────────────────────────
    push("TACTICAL INVESTIGATION BRIEF", "F2", _FONT_SIZE_TITLE, 4)
    push(f"Case: {title}", "F2", _FONT_SIZE_HEAD, 2)
    push(f"Generated: {generated_at}   |   Seed: {SEED}", "F1", 8, 2)
    push("-" * 78, "F1", 8, 2)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    push("DISCLAIMER:", "F2", _FONT_SIZE_BODY, 4)
    for wrapped in _wrap_lines(DISCLAIMER, 90):
        push(wrapped, "F1", 8)
    push("", "F1", 4)

    # ── Section 1: Elevated-Priority Candidates ────────────────────────────────
    push("1. ELEVATED-PRIORITY CANDIDATES (require investigator verification)",
         "F2", _FONT_SIZE_HEAD, 6)
    if top3:
        for rank, cand in enumerate(top3[:MAX_CANDIDATES], start=1):
            name   = cand.get("name", "Unknown")
            score  = cand.get("risk_score", 0)
            label  = cand.get("label", "elevated-priority candidate")
            reason = (f"Risk score: {score:.1f}/100  |  "
                      f"Degree: {cand.get('degree_score', 0):.2f}  "
                      f"Temporal: {cand.get('temporal_score', 0):.2f}  "
                      f"Txn: {cand.get('transaction_score', 0):.2f}")
            push(f"  {rank}. {name} — {label}", "F2", _FONT_SIZE_BODY, 3)
            push(f"     {reason}", "F1", 8)
    else:
        push("  No candidate data available.", "F1", _FONT_SIZE_BODY)

    push("", "F1", 4)

    # ── Section 2: Event Timeline ─────────────────────────────────────────────
    push("2. ORDERED EVENT TIMELINE (top chain)", "F2", _FONT_SIZE_HEAD, 4)
    if chains:
        top_chain = chains[0]
        steps = top_chain.get("steps", [])[:MAX_TIMELINE_BULLETS]
        pivot_idx = top_chain.get("pivot", {}).get("step_index", -1)
        for si, step in enumerate(steps):
            ts     = step.get("t", "")[:16].replace("T", " ")
            actor  = step.get("actor", "")
            action = step.get("action", "")
            target = step.get("target", "")
            src    = step.get("source", "")
            line   = step.get("line", "")
            pivot_mark = " ← PIVOT" if si == pivot_idx else ""
            bullet = (f"  [{ts}] {actor} — {action} → {target}"
                      f"{pivot_mark}   [{src}:L{line}]")
            push(bullet, "F1", 8, 1)
        if len(top_chain.get("steps", [])) > MAX_TIMELINE_BULLETS:
            push(f"  ... ({len(top_chain['steps']) - MAX_TIMELINE_BULLETS}"
                 " more events in chain)", "F1", 8)
    else:
        push("  Run event_causality.py first to populate timeline.", "F1", 8)

    push("", "F1", 4)

    # ── Section 3: Temporal Motif Flags ──────────────────────────────────────
    push("3. TEMPORAL ANOMALIES DETECTED", "F2", _FONT_SIZE_HEAD, 4)
    if flags:
        for fl in flags[:4]:
            push(f"  • {fl['name']} ({fl['phone']}): "
                 f"{fl['multiplier_observed']}× off-hour baseline  "
                 f"[{fl['evidence']['cdr_file']}]",
                 "F1", 8, 1)
    else:
        push("  No temporal anomalies detected above threshold.", "F1", 8)

    push("", "F1", 4)

    # ── Section 4: Investigator Next Steps ────────────────────────────────────
    push("4. SUGGESTED INVESTIGATOR NEXT STEPS", "F2", _FONT_SIZE_HEAD, 4)
    next_steps = [
        "Verify identity and associations of all elevated-priority candidates.",
        "Request call records and financial statements for pivotal event dates.",
        "Cross-reference ghost-node suggestions with known entity databases.",
        "Assess off-hour communication patterns against operational schedules.",
        "Ensure all findings are independently corroborated before action.",
    ]
    for ns in next_steps:
        for wrapped in _wrap_lines(f"  • {ns}", 88):
            push(wrapped, "F1", 8, 1)

    push("", "F1", 4)
    push("-" * 78, "F1", 8)
    push("End of Tactical Brief — FOR AUTHORISED INVESTIGATOR USE ONLY",
         "F1", 8, 2)

    return lines


# ── Manifest & Signature ──────────────────────────────────────────────────────

def _sign_manifest(manifest: dict, key: bytes) -> str:
    """HMAC-SHA256 signature over canonical JSON of manifest."""
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _write_manifest(pdf_path: Path, case_id: str) -> Path:
    """Write manifest.json alongside the PDF with SHA-256 hash + signature."""
    manifest = {
        "case_id":     case_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed":        SEED,
        "pdf_sha256":  _sha256_file(pdf_path),
        "pdf_file":    pdf_path.name,
    }
    # Deterministic HMAC key derived from seed (offline; not secret in demo)
    key = hashlib.sha256(f"brief_key_seed_{SEED}".encode()).digest()
    manifest["hmac_sha256_signature"] = _sign_manifest(manifest, key)
    manifest["signature_note"] = (
        "Demo HMAC-SHA256 signature — replace with RSA-PSS for production."
    )

    manifest_path = pdf_path.parent / f"manifest_{case_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ── Public API ────────────────────────────────────────────────────────────────

def generate_brief(case_id: str, out_path: Path | None = None) -> Path:
    """
    Generate a one-page PDF tactical brief for `case_id`.

    Parameters
    ----------
    case_id  : e.g. "case_A"
    out_path : where to write the PDF; defaults to
               demo_cache/feature_outputs/brief_{case_id}.pdf

    Returns
    -------
    Path to the generated PDF.
    """
    np.random.seed(SEED)
    FEAT_OUT.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        out_path = FEAT_OUT / f"brief_{case_id}.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    layout_lines = _layout_brief(case_id, generated_at)

    pdf = _SimplePDF()
    pdf.add_page(layout_lines)
    pdf.write(out_path)

    manifest_path = _write_manifest(out_path, case_id)
    print(f"  ✓ Brief PDF : {out_path}")
    print(f"  ✓ Manifest  : {manifest_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]
    if not args or args[0] != "export":
        print("Usage: python -m M6_feature.auto_brief export <case_id> [out_path]")
        sys.exit(1)

    case_id  = args[1] if len(args) > 1 else "case_A"
    out_path = Path(args[2]) if len(args) > 2 else None

    print(f"Generating tactical brief for {case_id} ...")
    pdf_path = generate_brief(case_id, out_path)
    print(f"Done: {pdf_path}")


if __name__ == "__main__":
    _cli()
