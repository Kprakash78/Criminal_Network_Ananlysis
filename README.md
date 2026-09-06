# AI-Powered Criminal Network Analysis System

This project is an end-to-end intelligence platform designed to extract, analyze, and query criminal networks from raw unstructured intelligence reports (like FIRs), Call Detail Records (CDRs), and financial transaction logs.

## Architecture Modules

The system is divided into six distinct, decoupled modules:

- **M1 (Data & NER):** Extracts structured entities (Persons, Phones, Locations, etc.) and basic co-occurrence relationships from raw documents using spaCy and Transformer models.
- **M2 (Knowledge Graph):** Builds a typed, semantic `NetworkX` graph from M1 outputs, assigning specific relationship types (e.g., `CALLED`, `TRANSFERRED_MONEY_TO`, `APPEARS_IN_CASE`).
- **M3 (Graph Analytics):** Analyzes the M2 graph to score entities for centrality (Betweenness, Degree) and identify suspicious behavioral patterns (e.g., Burner Phones, Smurfing, Structuring).
- **M4 (Local LLM & RAG):** Provides fully offline natural language summarization and Q&A using FAISS and local HuggingFace models, grounding responses securely in extracted evidence.
- **M5 (Agentic Orchestrator):** A LangGraph-based agent that orchestrates requests between the UI and the backend modules, maintaining context and handling follow-up queries.
- **M6 (Dashboard):** An interactive Streamlit application featuring PyVis network visualizations, entity tables, and an investigative chat interface.

## Setup & Installation

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Language Models:**
   You must download the base spaCy English model for M1's Named Entity Recognition to function:
   ```bash
   python -m spacy download en_core_web_sm
   ```
   *(Note: The transformer models for M4 will be downloaded automatically by HuggingFace the first time you run the system, but you must be connected to the internet for the initial pull.)*

## Usage

### 1. Generating the Initial Dataset
Before running the dashboard, you can build the initial static knowledge graph by running the offline M1 pipeline:
```bash
python M1/generate_dataset.py
python M1/pipeline.py
```
This will populate `M1/output/` with `entities.json` and `relationships.json`.

### 2. Running the Application
The primary interface is the M6 Streamlit dashboard. 

To run the application using the **Real Backend Modules** (rather than mock data), you must set the `CRIMINAL_USE_REAL_MODULES` environment variable to `1`.

**On Windows (PowerShell):**
```powershell
$env:CRIMINAL_USE_REAL_MODULES="1"; streamlit run M6/app.py
```

**On Linux/macOS:**
```bash
CRIMINAL_USE_REAL_MODULES=1 streamlit run M6/app.py
```

## New Features (v2.0)

### 🎯 Top-3 Suspect Ranking (`M3_feature/top3.py`)
Identifies the top 3 highest-risk entities using a transparent, weighted risk scoring formula:
```
risk_score = 0.20×degree + 0.20×temporal + 0.15×colocation + 0.20×call_pattern + 0.25×transaction
```
Each suspect includes plain-English explanations backed by exact evidence line references.

### 🔎 Case Search (`M6_feature/search_api.py`)
Semantic search across all case documents with exact `file:line` provenance for every finding. Uses sentence-transformers (all-MiniLM-L6-v2) with TF-IDF fallback for offline use.

### ⏱️ Timeline Player (`M6_feature/timeline_player.py`)
Animated playback of all case events (calls, transactions, FIR filings) in chronological order with network graph visualization. Includes Play/Pause/Speed controls.

### 📦 Evidence Exporter (`M6_feature/exporter.py`)
One-click court evidence ZIP packaging with:
- SHA-256 integrity hashes for every file
- RSA-2048 digital signature of manifest
- Optional AES-256 encryption
- Step-by-step verification guide

### Running the New Dashboard
```bash
# Quick start with demo data
chmod +x run_demo.sh && ./run_demo.sh

# Or manually
python3 demo_dataset/generate_demo_outputs.py
python3 -m M3_feature.top3
CRIMINAL_USE_REAL_MODULES=0 streamlit run M6_feature/search_ui.py --server.headless true
```

See [demo_instructions.md](demo_instructions.md) for the full 5-minute demo walkthrough.

## Security & Privacy
This application is explicitly designed for secure, offline environments. All graph analytics (M3) and LLM summarization (M4) run locally. No case data is transmitted to external API providers like OpenAI or Anthropic during analysis.

### Data Handling
- **All data in this repo is synthetic** — see [demo_dataset/SYNTHETIC_DATA_NOTICE.md](demo_dataset/SYNTHETIC_DATA_NOTICE.md)
- **DO NOT commit real FIRs/CDRs** — see [PII_POLICY.md](PII_POLICY.md)
- **No API keys required** — see [SECRETS_README.md](SECRETS_README.md)
- Evidence packages are **cryptographically signed** for integrity verification
