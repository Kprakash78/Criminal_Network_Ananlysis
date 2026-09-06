# 5-Minute Demo — Criminal Network Analysis System

## Prerequisites

- Python 3.10+ installed
- Terminal access

## Quick Start

```bash
chmod +x run_demo.sh
./run_demo.sh
```

The script will automatically set up everything and open the dashboard at `http://localhost:8501`.

---

## Demo Walkthrough (5 minutes)

### Minute 1: Overview & Case Selection (0:00 - 1:00)

1. Open `http://localhost:8501` in your browser
2. Notice the **sidebar** showing system status — all components green ✅
3. The **demo_case** is automatically selected (synthetic data)
4. Point out: "All data is synthetic. No real FIRs or CDRs are used."

### Minute 2: Case Search (1:00 - 2:00)

1. Click the **🔎 Case Search** tab
2. Type: `"Where was Ravi Kumar seen?"`
3. Click **🔍 Search**
4. Show the **Answer** box — notice inline citations like `[source: data/firs/FIR_001.txt:L6-L8]`
5. Expand a search result → click **📄 View Source File** to see the highlighted lines
6. Point out: "Every answer points to exactly where it came from — no hallucinations"

### Minute 3: Top-3 Suspects (2:00 - 3:00)

1. Click the **🎯 Top-3 Suspects** tab
2. Show the three suspect cards:
   - **Risk Score** (0-100) with gradient coloring
   - **3-bullet explanation** with evidence references
   - **Evidence list** with clickable file:line links
3. Expand **📊 Scoring Methodology** to show the formula
4. Point out: "Every score is transparent and traceable to specific evidence"

### Minute 4: Timeline Player (3:00 - 4:00)

1. Click the **⏱️ Timeline Player** tab
2. Click **▶ Play** to start the animation
3. Watch events appear chronologically on the network graph:
   - 📞 Calls highlighted in **blue**
   - 💰 Transactions in **orange**
   - 📄 FIR filings in **red**
4. Use **Speed** dropdown to change to **5x** for faster playback
5. Click **⏭ Next** to step through events manually
6. Point out: "Judges can see the full chronology with source references"

### Minute 5: Evidence Export (4:00 - 5:00)

1. Click the **📦 Export Evidence** tab
2. Review the package contents table
3. Click the **🔴 Export Evidence Package** button
4. Wait for "✅ Evidence package created"
5. Click **⬇️ Download Evidence ZIP**
6. Show the ZIP contents:
   - `manifest.json` — SHA-256 hashes of every file
   - `manifest.sig` — RSA-2048 digital signature
   - `VERIFY.md` — step-by-step verification instructions
7. Point out: "This package is cryptographically signed and independently verifiable"

---

## Key Talking Points

### For Judges
- **Every claim is traceable**: Click any citation to see the exact source line
- **No hallucinations**: Template-based answers grounded in actual documents
- **Offline-capable**: Works on a laptop without internet connection
- **Tamper-evident**: SHA-256 hashes + RSA signatures in evidence packages

### For Technical Reviewers
- **Transparent scoring**: 5-component weighted formula, all weights configurable
- **Provenance tracking**: Every retrieval is logged to `retrieval_logs.jsonl`
- **Fallback design**: TF-IDF when no ML model available, HMAC when no RSA library
- **Deterministic demo**: Same data → same results every time (seeded randomness)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `streamlit: command not found` | Run `pip install streamlit` |
| `ModuleNotFoundError` | Ensure you're in the repo root and venv is activated |
| Search returns low scores | Normal for TF-IDF fallback; sentence-transformers improves quality |
| Empty timeline | Check that `data/cdrs/cdr.csv` exists |
| Export fails | Ensure `demo_dataset/M1_output/` was generated |
