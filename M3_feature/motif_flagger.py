"""
M3_feature/motif_flagger.py
PS 26152 — AI-Powered Criminal Network Analysis System
======================================================
Temporal Motif Flagger — Unusual Time-Window Detection.

Rule-based detection of anomalous call-time distributions per phone number.
For each number, computes an hourly call histogram and flags numbers whose
off-hour call density exceeds a configurable multiple of the baseline.

Off-hour window: 23:00–04:59 (configurable).
Baseline: average calls per hour during peak hours (08:00–22:59).
Flag if off-hour rate > OFF_HOUR_MULTIPLIER × baseline.

OUTPUT: M3_feature/motif_flags.json
        demo_cache/feature_outputs/motif_flags.json

LANGUAGE: Flags are presented as "unusual timing pattern — requires
investigator verification". No guilt determinations are made.

Usage (CLI):
    python -m M3_feature.motif_flagger flag case_A
    python -m M3_feature.motif_flagger flag_all

API:
    from M3_feature.motif_flagger import flag_motifs
    flags = flag_motifs("case_A")
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEMO_DIR    = REPO_ROOT / "demo_dataset"
OUTPUT_DIR  = Path(__file__).resolve().parent
FEAT_OUT    = REPO_ROOT / "demo_cache" / "feature_outputs"

SEED = 42
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
OFF_HOURS_START  = 23   # 23:00 inclusive
OFF_HOURS_END    = 5    # 05:00 exclusive (so hours 23, 0, 1, 2, 3, 4)
OFF_HOUR_MULTIPLIER = 1.5   # flag if off-hour rate > 1.5× peak-hour baseline

PERSONS = [
    {"name": "Ravi Kumar",     "phone": "9876543210"},
    {"name": "Sunita Sharma",  "phone": "9123456789"},
    {"name": "Mohammed Iqbal", "phone": "9988776655"},
    {"name": "Priya Nair",     "phone": "8877665544"},
    {"name": "Deepak Verma",   "phone": "7766554433"},
    {"name": "Anjali Singh",   "phone": "9654321098"},
    {"name": "Rakesh Yadav",   "phone": "9543210987"},
    {"name": "Fatima Begum",   "phone": "9432109876"},
    {"name": "Suresh Patil",   "phone": "9321098765"},
    {"name": "Kavya Reddy",    "phone": "9210987654"},
]
PHONE_TO_NAME = {p["phone"]: p["name"] for p in PERSONS}

_OFF_HOURS = set(range(OFF_HOURS_START, 24)) | set(range(0, OFF_HOURS_END))
_PEAK_HOURS = set(range(24)) - _OFF_HOURS


def _is_off_hour(hour: int) -> bool:
    return hour in _OFF_HOURS


def flag_motifs(case_id: str,
                multiplier: float = OFF_HOUR_MULTIPLIER) -> list[dict]:
    """
    Analyse hourly call distribution for every phone in `case_id`'s CDR.
    Returns list of flag dicts for numbers with anomalous off-hour patterns.

    Each flag:
      {
        "phone": "...", "name": "...",
        "off_hour_calls": 5, "peak_hour_calls": 3,
        "off_hour_rate": 0.83,  "baseline_rate": 0.20,
        "multiplier_observed": 4.15,
        "flagged_hours": [0, 1, 2, 23],
        "explanation": "...",
        "evidence": {"cdr_file": "...", "flagged_rows": [3, 7, 12]}
      }
    """
    np.random.seed(SEED)

    gt_path = DEMO_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError("ground_truth.json missing")
    gt = json.loads(gt_path.read_text())[case_id]

    cdr_path = REPO_ROOT / gt["cdr_file"]
    if not cdr_path.exists():
        return []

    # hour_counts[phone][hour] = n_calls; flagged_rows[phone] = [line_nos]
    hour_counts: dict = defaultdict(lambda: defaultdict(int))
    flagged_rows: dict = defaultdict(list)

    with open(cdr_path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            phone = row.get("caller", "").strip()
            hour  = int(row.get("hour", row.get("timestamp", " 12:")[11:13]))
            if phone:
                hour_counts[phone][hour] += 1
                if _is_off_hour(hour):
                    flagged_rows[phone].append(line_no)

    flags = []
    for phone, h_map in hour_counts.items():
        off_calls  = sum(h_map.get(h, 0) for h in _OFF_HOURS)
        peak_calls = sum(h_map.get(h, 0) for h in _PEAK_HOURS)
        total      = off_calls + peak_calls
        if total == 0:
            continue

        n_off  = len(_OFF_HOURS)  # 7 hours
        n_peak = len(_PEAK_HOURS) # 17 hours

        off_rate      = off_calls  / n_off
        baseline_rate = peak_calls / n_peak if n_peak > 0 else 0

        observed_mult = off_rate / baseline_rate if baseline_rate > 0 else float(off_calls)

        if observed_mult < multiplier:
            continue

        active_off_hours = sorted(h for h in _OFF_HOURS if h_map.get(h, 0) > 0)
        name = PHONE_TO_NAME.get(phone, phone)

        explanation = (
            f"Phone {phone} ({name}) shows {off_calls} call(s) during "
            f"off-hours ({OFF_HOURS_START}:00–{OFF_HOURS_END}:00) vs "
            f"{peak_calls} during peak hours — "
            f"{observed_mult:.1f}× the baseline rate. "
            "This pattern requires investigator verification."
        )

        flags.append({
            "phone": phone,
            "name":  name,
            "off_hour_calls":      off_calls,
            "peak_hour_calls":     peak_calls,
            "off_hour_rate":       round(off_rate, 4),
            "baseline_rate":       round(baseline_rate, 4),
            "multiplier_observed": round(observed_mult, 3),
            "flagged_hours":       active_off_hours,
            "explanation":         explanation,
            "evidence": {
                "cdr_file":    str(cdr_path.relative_to(REPO_ROOT)),
                "flagged_rows": flagged_rows[phone][:20],   # cap for readability
            },
        })

    flags.sort(key=lambda f: f["multiplier_observed"], reverse=True)
    return flags


def flag_all(multiplier: float = OFF_HOUR_MULTIPLIER) -> dict[str, list[dict]]:
    """Flag motifs for every case in ground_truth.json."""
    gt = json.loads((DEMO_DIR / "ground_truth.json").read_text())
    return {case_id: flag_motifs(case_id, multiplier=multiplier) for case_id in gt}


def save_flags(flags_by_case: dict) -> Path:
    """Write motif_flags.json to M3_feature/ and feature_outputs/."""
    data = {
        "seed": SEED,
        "off_hours": f"{OFF_HOURS_START}:00-{OFF_HOURS_END}:00",
        "multiplier_threshold": OFF_HOUR_MULTIPLIER,
        "flags": flags_by_case,
    }
    out1 = OUTPUT_DIR / "motif_flags.json"
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    out2 = FEAT_OUT / "motif_flags.json"
    for path in (out1, out2):
        path.write_text(json.dumps(data, indent=2))
    return out1


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m M3_feature.motif_flagger flag <case_id>")
        print("       python -m M3_feature.motif_flagger flag_all")
        sys.exit(1)

    cmd = args[0]
    if cmd == "flag":
        case_id = args[1] if len(args) > 1 else "case_A"
        print(f"Flagging temporal motifs for {case_id} ...")
        flags = flag_motifs(case_id)
        save_flags({case_id: flags})
        print(f"  {len(flags)} flag(s).")
        for fl in flags:
            print(f"  {fl['name']} ({fl['phone']}): "
                  f"{fl['multiplier_observed']}× off-hour baseline")
    elif cmd == "flag_all":
        print("Flagging temporal motifs for all cases ...")
        all_f = flag_all()
        save_flags(all_f)
        total = sum(len(v) for v in all_f.values())
        print(f"  {total} flags across {len(all_f)} cases.")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
