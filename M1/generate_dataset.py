"""
M1.1 — Synthetic Dataset Generator
PS 26152 — AI-Powered Criminal Network Analysis System

Produces:
  - data/firs/FIR_<n>.txt        : ≥30 realistic FIR text files (English + Hinglish)
  - data/cdrs/cdr.csv             : ≥100 CDR (call detail record) rows
  - data/transactions/trans.csv   : ≥50 transaction record rows

Entity overlap is deliberate: the same person/phone/vehicle appears under
different name forms across documents so the resolver has real work to do.
"""

import csv
import os
import random
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed for reproducibility (same input → same output, per NFR requirement)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Shared entity pool — these entities will deliberately appear across
# multiple documents under variant names to exercise the resolver.
# ---------------------------------------------------------------------------

# Format: (canonical_name, [aliases], phone, vehicle, account)
PERSONS = [
    ("Ravi Kumar",       ["Ravi K.", "R. Kumar", "Ravi"],          "9876543210", "DL01AB1234", "ACC00101"),
    ("Sunita Sharma",    ["S. Sharma", "Sunita S."],               "9123456789", "MH12CD5678", "ACC00102"),
    ("Mohammed Iqbal",   ["M. Iqbal", "Mohd Iqbal", "Iqbal"],      "9988776655", "UP32EF9012", "ACC00103"),
    ("Priya Nair",       ["P. Nair", "Priya N."],                  "8877665544", "KA05GH3456", "ACC00104"),
    ("Deepak Verma",     ["D. Verma", "Deepak V.", "Deepak"],      "7766554433", "RJ14IJ7890", "ACC00105"),
    ("Anjali Singh",     ["A. Singh", "Anjali S."],                "9654321098", "GJ01KL2345", "ACC00106"),
    ("Rakesh Yadav",     ["R. Yadav", "Rakesh Y."],                "9543210987", "HR26MN6789", "ACC00107"),
    ("Fatima Begum",     ["F. Begum", "Fatima B."],                "9432109876", "TN09OP0123", "ACC00108"),
    ("Suresh Patil",     ["S. Patil", "Suresh P."],                "9321098765", "MP07QR4567", "ACC00109"),
    ("Kavya Reddy",      ["K. Reddy", "Kavya R."],                 "9210987654", "AP28ST8901", "ACC00110"),
]

LOCATIONS = [
    "Lajpat Nagar, Delhi",
    "Bandra West, Mumbai",
    "Connaught Place, Delhi",
    "Koramangala, Bengaluru",
    "Sector 17, Chandigarh",
    "Park Street, Kolkata",
    "T. Nagar, Chennai",
    "Hazratganj, Lucknow",
    "Camp Area, Pune",
    "Ellis Bridge, Ahmedabad",
]

ORGANIZATIONS = [
    "Sunrise Finance Ltd.",
    "Metro Logistics Pvt. Ltd.",
    "Sharma & Sons Traders",
    "Al-Barkat Enterprises",
    "Reddy Construction Co.",
]

CASE_NUMBERS = [f"FIR{str(i).zfill(3)}" for i in range(101, 160)]

# ---------------------------------------------------------------------------
# Hinglish sentence templates (code-mixed Hindi–English)
# ---------------------------------------------------------------------------

HINGLISH_SENTENCES = [
    "{name} ne bataya ki uska mobile {phone} par call aaya tha.",
    "Witness ne dekha ki {name} gaadi {vehicle} mein baith ke bhaag gaya.",
    "{name} ko {location} mein pakda gaya, uske paas cash tha.",
    "Chor {name} ne {org} ka maal chura liya.",
    "Police ne {name} ko {location} se giraftaar kiya.",
    "{name} ka account number {account} suspect hai.",
    "Aankhon dekha gawah ne kaha ki {name} aur {alias} dono wahan the.",
    "{name} ki gaadi {vehicle} CCTV mein pakdi gayi {location} ke paas.",
    "Call record se pata chala ki {phone} se {phone2} par baar baar call hua.",
    "{name} ne {org} ke through paisa bheja, account {account} mein.",
]

# ---------------------------------------------------------------------------
# English FIR template blocks
# ---------------------------------------------------------------------------

ENGLISH_INTRO_TEMPLATES = [
    "This First Information Report is registered on the complaint of {complainant}, "
    "resident of {location}. The complainant states that on {date}, "
    "the accused {name} was seen near {location2}.",

    "On {date}, a complaint was received at Police Station regarding an incident "
    "involving {name}, alias {alias}, in the area of {location}.",

    "Vide complaint no. {case}, dated {date}, the undersigned officer has registered "
    "this FIR against {name} (hereinafter 'accused') for offences under relevant IPC sections.",

    "A written complaint submitted by {complainant} alleges that {name} "
    "and associates operating from {location} have been involved in financial fraud "
    "using account {account}.",
]

ENGLISH_DETAIL_TEMPLATES = [
    "The accused {name} contacted the complainant via phone number {phone} "
    "and demanded money under false pretenses. The accused was last seen "
    "driving vehicle {vehicle} near {location}.",

    "Investigations revealed that {name} is associated with {org} "
    "and has been receiving suspicious funds in account {account}. "
    "The accused operates from {location}.",

    "CCTV footage shows vehicle {vehicle} parked outside the crime scene at {location}. "
    "The registered owner of the vehicle is {name}.",

    "Phone records indicate repeated calls between {phone} and {phone2} "
    "during the period of the alleged offence. The subscriber of {phone} "
    "is identified as {name}.",

    "The accused {name}, also known as {alias}, was apprehended at {location}. "
    "During search, a mobile phone bearing SIM with number {phone} was recovered.",
]

ENGLISH_CLOSING_TEMPLATES = [
    "Action is being taken as per law. Further investigation is in progress. "
    "This FIR is registered and case diary is being maintained.",

    "The matter is under active investigation. "
    "Forensic analysis of recovered items is pending.",

    "Accused has been remanded to judicial custody. "
    "Further details will be added upon completion of investigation.",
]


def _rand_date(start_year: int = 2024, end_year: int = 2026) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 8, 1)
    delta = end - start
    rnd = start + timedelta(days=random.randint(0, delta.days))
    return rnd.strftime("%d-%m-%Y")


def _rand_ts(start_year: int = 2024, end_year: int = 2026) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 8, 1)
    delta = end - start
    rnd = start + timedelta(
        days=random.randint(0, delta.days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return rnd.strftime("%Y-%m-%dT%H:%M:%S")


def _pick(lst):
    return random.choice(lst)


def _alias(person_idx: int) -> str:
    person = PERSONS[person_idx]
    return _pick(person[1])  # pick a non-canonical alias


# ---------------------------------------------------------------------------
# FIR generation
# ---------------------------------------------------------------------------

def _generate_fir(fir_num: int, primary_person_idx: int, secondary_person_idx: int | None) -> str:
    p = PERSONS[primary_person_idx]
    s = PERSONS[secondary_person_idx] if secondary_person_idx is not None else None

    loc1 = _pick(LOCATIONS)
    loc2 = _pick([l for l in LOCATIONS if l != loc1])
    org = _pick(ORGANIZATIONS)
    date = _rand_date()
    case = _pick(CASE_NUMBERS)

    # Choose whether to use canonical name or an alias for variety
    # (this creates the overlap the resolver must handle)
    use_alias = random.random() < 0.45
    display_name = _alias(primary_person_idx) if use_alias else p[0]

    intro = _pick(ENGLISH_INTRO_TEMPLATES).format(
        complainant=_pick([q[0] for q in PERSONS if q != p]),
        location=loc1,
        name=display_name,
        alias=_alias(primary_person_idx),
        date=date,
        case=case,
        account=p[4],
        location2=loc2,
    )

    detail = _pick(ENGLISH_DETAIL_TEMPLATES).format(
        name=display_name,
        alias=_alias(primary_person_idx),
        phone=p[2],
        phone2=(s[2] if s else _pick([q[2] for q in PERSONS if q != p])),
        vehicle=p[3],
        location=loc1,
        account=p[4],
        org=org,
    )

    closing = _pick(ENGLISH_CLOSING_TEMPLATES)

    # Inject 2–4 Hinglish sentences
    n_hinglish = random.randint(2, 4)
    hinglish_block_lines = []
    for _ in range(n_hinglish):
        tmpl = _pick(HINGLISH_SENTENCES)
        phone2_candidate = s[2] if s else _pick([q[2] for q in PERSONS if q != p])
        line = tmpl.format(
            name=display_name,
            alias=_alias(primary_person_idx),
            phone=p[2],
            phone2=phone2_candidate,
            vehicle=p[3],
            location=_pick(LOCATIONS),
            account=p[4],
            org=org,
        )
        hinglish_block_lines.append(line)
    hinglish_block = "\n".join(hinglish_block_lines)

    # Add secondary person reference if available
    secondary_block = ""
    if s:
        use_alias_s = random.random() < 0.5
        display_s = _alias(secondary_person_idx) if use_alias_s else s[0]
        secondary_block = (
            f"Further, it is reported that {display_s} (phone: {s[2]}) "
            f"was in contact with the accused and was seen at {_pick(LOCATIONS)}."
        )

    fir_text = textwrap.dedent(f"""\
        FIRST INFORMATION REPORT
        Case No.: {case}
        Date: {date}
        Police Station: [Synthetic]

        {intro}

        {detail}

        {hinglish_block}

        {secondary_block}

        {closing}
    """)
    return fir_text


def generate_firs(output_dir: Path, count: int = 35) -> list[Path]:
    """Generate `count` FIR text files with deliberate entity overlap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n_persons = len(PERSONS)

    for i in range(count):
        primary_idx = i % n_persons
        # Pair with a secondary person ~60% of the time
        secondary_idx = (i + 3) % n_persons if random.random() < 0.6 else None

        fir_text = _generate_fir(i + 1, primary_idx, secondary_idx)
        fname = output_dir / f"FIR_{str(i + 1).zfill(3)}.txt"
        fname.write_text(fir_text, encoding="utf-8")
        paths.append(fname)

    return paths


# ---------------------------------------------------------------------------
# CDR generation (≥100 rows)
# ---------------------------------------------------------------------------

def generate_cdrs(output_dir: Path, count: int = 120) -> Path:
    """Generate a CDR CSV with `count` rows plus one planted communication spike."""
    output_dir.mkdir(parents=True, exist_ok=True)
    phones = [p[2] for p in PERSONS]
    out_path = output_dir / "cdr.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["caller", "callee", "timestamp", "duration_seconds", "tower_location"])
        for _ in range(count):
            caller = _pick(phones)
            callee = _pick([p for p in phones if p != caller])
            writer.writerow([
                caller,
                callee,
                _rand_ts(),
                random.randint(10, 600),
                _pick(LOCATIONS),
            ])

        # Planted communication-spike pattern so M3's COMMUNICATION_SPIKE rule has
        # a real signal to detect: a burst of calls between two phones in a short window.
        spike_caller = phones[0]
        spike_callee = phones[1]
        spike_base = datetime(2026, 6, 15, 9, 0, 0)
        planted_spike_count = 6
        for i in range(planted_spike_count):
            ts = spike_base + timedelta(minutes=i * 15)
            writer.writerow([
                spike_caller,
                spike_callee,
                ts.strftime("%Y-%m-%dT%H:%M:%S"),
                120,
                "Bandra West, Mumbai",
            ])

    return out_path


# ---------------------------------------------------------------------------
# Transaction record generation (≥50 rows)
# ---------------------------------------------------------------------------

def generate_transactions(output_dir: Path, count: int = 65) -> Path:
    """Generate a transaction CSV with `count` rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    accounts = [p[4] for p in PERSONS]
    out_path = output_dir / "transactions.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_account", "target_account", "amount_inr", "timestamp", "remarks"])
        for _ in range(count):
            src = _pick(accounts)
            tgt = _pick([a for a in accounts if a != src])
            amount = round(random.uniform(500, 200000), 2)
            writer.writerow([
                src,
                tgt,
                amount,
                _rand_ts(),
                _pick(["Transfer", "Payment", "Loan", "Commission", "Settlement", "Unknown"]),
            ])

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(base_dir: Path | str = ".") -> dict:
    """
    Generate the full synthetic dataset under `base_dir/data/`.

    Returns a dict with paths and counts for verification.
    """
    # Reset seed here so every call to generate_all() is deterministic
    # regardless of what random calls preceded it in the same process.
    random.seed(RANDOM_SEED)
    base = Path(base_dir)
    fir_dir = base / "data" / "firs"
    cdr_dir = base / "data" / "cdrs"
    trans_dir = base / "data" / "transactions"

    fir_paths = generate_firs(fir_dir, count=35)
    cdr_path = generate_cdrs(cdr_dir, count=120)
    trans_path = generate_transactions(trans_dir, count=65)

    return {
        "fir_dir": str(fir_dir),
        "fir_count": len(fir_paths),
        "cdr_path": str(cdr_path),
        "transaction_path": str(trans_path),
        "persons": [p[0] for p in PERSONS],
        "locations": LOCATIONS,
        "organizations": ORGANIZATIONS,
    }


if __name__ == "__main__":
    import json
    result = generate_all(base_dir=Path(__file__).parent)
    print(json.dumps(result, indent=2))
