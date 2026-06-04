"""
src/rag/pipeline.py
--------------------
End-to-end RAG pipeline: query → retrieve → generate → return.
"""

import time
from typing import Dict, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.rag.retriever import MedicalRetriever
from src.rag.generator import generate_answer


class MedRAGPipeline:
    """
    Full RAG pipeline for medical question answering.
    
    Usage:
        pipeline = MedRAGPipeline()
        pipeline.load()
        result = pipeline.query("What are symptoms of diabetes?")
    """

    def __init__(self):
        self.retriever = MedicalRetriever()
        self.is_loaded = False

    def load(self) -> bool:
        """Load vector store from disk. Returns True if successful."""
        success = self.retriever.load()
        if success:
            self.is_loaded = True
            print(f"✅ MedRAG loaded: {len(self.retriever.chunks)} chunks indexed")
        else:
            print("⚠️  Vector store not found. Run: python src/data/build_index.py")
        return success

    def query(self, question: str, top_k: int = None) -> Dict:
        """
        Run the full RAG pipeline on a question.
        
        Args:
            question: user's medical question
            top_k: number of chunks to retrieve
        
        Returns:
            dict with answer, sources, retrieved chunks, latency, scores
        """
        if not self.is_loaded:
            return {
                "question": question,
                "answer": "Knowledge base not loaded. Please run: python src/data/build_index.py",
                "sources": [],
                "chunks": [],
                "latency_ms": 0,
                "retrieval_scores": [],
            }

        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        start_time = time.time()

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.search(question, top_k=top_k)

        # Step 2: Generate answer from context
        generation_result = generate_answer(question, chunks)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": generation_result["answer"],
            "sources": generation_result["sources"],
            "chunks": chunks,
            "latency_ms": round(latency_ms, 1),
            "retrieval_scores": [c["similarity_score"] for c in chunks],
            "model": generation_result["model"],
            "num_chunks_retrieved": len(chunks),
        }


# Module-level singleton
_pipeline_instance = None

def get_pipeline() -> MedRAGPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MedRAGPipeline()
        _pipeline_instance.load()
    return _pipeline_instance
