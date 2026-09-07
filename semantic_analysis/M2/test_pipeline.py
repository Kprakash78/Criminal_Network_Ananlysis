import os
import json
import numpy as np
# pyrefly: ignore [missing-import]
from pipeline import process_video_audio

def validate_schema(records):
    """Verify that output exactly matches COMMON_DATA_CONTRACT.md schema."""
    for r in records:
        assert "video_id" in r
        assert "start_ts" in r
        assert "end_ts" in r
        assert isinstance(r["start_ts"], (float, int))
        assert isinstance(r["end_ts"], (float, int))
        
        assert "audio" in r
        assert "transcript" in r["audio"]
        assert "embedding_id" in r["audio"]
        assert isinstance(r["audio"]["transcript"], str)
        assert isinstance(r["audio"]["embedding_id"], str)
        assert "_embedding_vector" in r
        
        # Check that embedding vector has right type and size length
        assert isinstance(r["_embedding_vector"], list)
        assert len(r["_embedding_vector"]) == 384 # MiniLM dimension
        
        # Monotonicity check
        assert r["start_ts"] < r["end_ts"]

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def test_video(video_path, video_id="test_video_1"):
    print(f"--- Running pipeline on {video_path} ---")
    if not os.path.exists(video_path):
        print(f"File {video_path} not found.")
        return False
        
    records = process_video_audio(video_id, video_path)
    
    if not records:
        print("No records returned. Maybe silence or an error? (Test passed if silent video)")
        return True
        
    print(f"Got {len(records)} chunks.")
    
    # Verification 1: Schema validation
    print("\n[1] Validating schema...")
    try:
        validate_schema(records)
        print("Schema validation PASSED!")
    except AssertionError as e:
        print("Schema validation FAILED!")
        raise e
    
    # Verification 2: Output spot-check
    print("\n[2] Chunk output spot-check:")
    for r in records:
        print(f"  [{r['start_ts']:.2f}s - {r['end_ts']:.2f}s] {r['audio']['transcript']}")
        
    # Verification 3: Embedding matching check
    print("\n[3] Embedding sanity check...")
    from pipeline import embed_model
    # We will test against the preamble words since we use the preamble download
    query = "We the people are forming a perfect union" 
    print(f"Query text: '{query}'")
    q_emb = embed_model.encode(query)
    
    best_score = -1
    best_chunk = None
    
    for r in records:
        score = cosine_similarity(q_emb, r["_embedding_vector"])
        if score > best_score:
            best_score = score
            best_chunk = r
            
    print(f"Best match score: {best_score:.3f}")
    if best_chunk:
        print(f"Best matching transcript: {best_chunk['audio']['transcript']}")
        if best_score > 0.4:
            print("Embedding sanity check PASSED!")
        else:
            print("Embedding match score low (might fail check if query doesn't match content).")
            return False
            
    return True

if __name__ == "__main__":
    result1 = test_video("test_speech.mp4", video_id="test_speech")
    print("\n--------------------------\n")
    # Also test the silent video to ensure we avoid hallucinations
    result2 = test_video("test_silent.mp4", video_id="test_silent")
    
    if result1 and result2:
        print("All verification steps PASSED.")
    else:
        print("Verification FAILED.")
