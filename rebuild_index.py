from M4.pipeline import RAGPipeline

print("Rebuilding FAISS Index...")
pipeline = RAGPipeline()
pipeline.load(rebuild_index=True)
print("Index rebuild complete.")
