import json
from M5.entry import handle_investigator_request, NewCaseUpload
from M2.query import CriminalGraph
from M4.pipeline import RAGPipeline
import asyncio

cases = [
    ("dummy_high_1.txt", "FIR_HIGH"),
    ("dummy_medium.txt", "FIR_MED"),
    ("dummy_low_1.txt", "FIR_LOW")
]

from M2.graph_builder import load_graph_from_m1_output
graph = load_graph_from_m1_output("M1/output/entities.json", "M1/output/relationships.json")
m4 = RAGPipeline()
m4.load(graph=graph)

async def run_all():
    results = {}
    for filename, case_id in cases:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        
        req = NewCaseUpload(case_id=case_id, case_text=text)
        res = handle_investigator_request(f"TEST_{case_id}", req, use_mocks=False, graph=graph, m4_pipeline=m4, cdr_path="M1/data/cdrs/cdr.csv", txn_path="M1/data/transactions/transactions.csv")
        results[case_id] = {
            "confidence": res.confidence,
            "evidence": res.evidence,
            "summary": res.response_text
        }
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done")

if __name__ == "__main__":
    asyncio.run(run_all())
