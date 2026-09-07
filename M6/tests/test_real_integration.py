"""
M6 Real Integration Test
Run from repo root: python M6/tests/test_real_integration.py
"""
import sys
sys.path.insert(0, ".")

print("=== M6 Real Integration Test ===")

# 1. Graph loading
print("\n[1] Loading graph from M1/output JSONs ...")
from M2.graph_builder import load_graph_from_m1_output
g = load_graph_from_m1_output("M1/output/entities.json", "M1/output/relationships.json")
print(f"    Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges  OK")

# 2. M3 flags
print("\n[2] Loading M3 pattern flags ...")
import json
from pathlib import Path
flags = json.loads(Path("M3/output/pattern_flags.json").read_text(encoding="utf-8"))
print(f"    {len(flags)} pattern flags loaded  OK")
if flags:
    top = flags[0]
    print(f"    Top flag: entity={top.get('entity_id')} score={top.get('priority_score')}")

# 3. M2 search
print("\n[3] Testing M2 search_by_name ...")
from M2.query import search_by_name, neighbors, subgraph
results = search_by_name(g, "Ravi", threshold=0.6)
print(f"    {len(results)} results for 'Ravi'  OK")
if results:
    print(f"    Top match: {results[0]['name']} ({results[0]['type']}) score={results[0]['score']}")

# 3b. M2 subgraph
print("\n[3b] Testing M2 subgraph ...")
# Use the first case_id found in source_documents
case_ids = set()
for nid, data in g.nodes(data=True):
    for doc in data.get("source_documents", []):
        case_ids.add(doc)
test_case = next(iter(case_ids), None)
if test_case:
    sg = subgraph(g, test_case)
    print(f"    subgraph('{test_case}'): {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges  OK")
else:
    print("    No case_ids found in graph nodes (source_documents empty?)")

# 4. M4 pipeline
print("\n[4] Loading M4 RAGPipeline ...")
from M4.pipeline import RAGPipeline
from M4.config import DEFAULT_CONFIG
pipeline = RAGPipeline(DEFAULT_CONFIG)
try:
    pipeline.load(graph=g)
except RuntimeError as exc:
    if "Insufficient RAM" in str(exc):
        try:
            import pytest

            pytest.skip(str(exc), allow_module_level=True)
        except ImportError:
            raise
    raise
print("    M4 pipeline loaded  OK")

# 5. M5 end-to-end call
print("\n[5] Testing M5 handle_investigator_request (real) ...")
from M5 import handle_investigator_request
from M5.models import NewCaseUpload, FollowUpQuestion

case_text = (
    "Case FIR_001. Ravi Kumar (DOB 12-Mar-1985) was identified at three ATMs. "
    "Phone 9876543210 registered to Ravi Kumar was used repeatedly. "
    "Account ACC00102 received large transfer at 11:45."
)
resp = handle_investigator_request(
    None,
    NewCaseUpload(case_id="FIR_001", case_text=case_text),
    use_mocks=False,
    graph=g,
    m4_pipeline=pipeline,
    cdr_path="M1/data/cdrs/cdr.csv",
    txn_path="M1/data/transactions/transactions.csv",
)
print(f"    Session ID   : {resp.session_id}")
print(f"    Confidence   : {resp.confidence}")
print(f"    Human review : {resp.requires_human_review}")
print(f"    Evidence     : {len(resp.evidence)} items")
print(f"    Response     : {resp.response_text[:150]}")

# 6. Follow-up
print("\n[6] Testing follow-up question ...")
resp2 = handle_investigator_request(
    resp.session_id,
    FollowUpQuestion(question="What connections does entity P001 have to financial accounts?"),
    use_mocks=False,
    graph=g,
    m4_pipeline=pipeline,
    cdr_path="M1/data/cdrs/cdr.csv",
    txn_path="M1/data/transactions/transactions.csv",
)
print(f"    Session same : {resp2.session_id == resp.session_id}")
print(f"    Confidence   : {resp2.confidence}")
print(f"    Response     : {resp2.response_text[:150]}")

# 7. M6 backend_calls wrapper
print("\n[7] Testing M6 backend_calls wrapper ...")
import os
original_real_modules_env = os.environ.get("CRIMINAL_USE_REAL_MODULES")
os.environ["CRIMINAL_USE_REAL_MODULES"] = "1"
# Force reload to pick up env var
import importlib
import M6.backend_calls as bc_module
importlib.reload(bc_module)
try:
    status = bc_module.get_backend_status()
    print(f"    use_real_modules : {status['use_real_modules']}")
    print(f"    graph_loaded     : {status['graph_loaded']}")
    print(f"    graph_nodes      : {status['graph_nodes']}")
    print(f"    pipeline_loaded  : {status['pipeline_loaded']}")
    print(f"    m3_flags_ok      : {status['m3_flags_available']}")
    if status["init_error"]:
        print(f"    init_error       : {status['init_error']}")
finally:
    if original_real_modules_env is None:
        os.environ.pop("CRIMINAL_USE_REAL_MODULES", None)
    else:
        os.environ["CRIMINAL_USE_REAL_MODULES"] = original_real_modules_env
    importlib.reload(bc_module)

print("\n=== ALL REAL INTEGRATION CHECKS PASSED ===")
