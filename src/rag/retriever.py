"""
src/rag/retriever.py
---------------------
Builds and queries a FAISS vector store for semantic search.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.rag.embedder import embed_texts, embed_query


class MedicalRetriever:
    """
    FAISS-based semantic retriever for medical text chunks.
    
    Usage:
        retriever = MedicalRetriever()
        retriever.build(chunks)          # Build index from chunks
        retriever.save()                 # Save to disk
        retriever.load()                 # Load from disk
        results = retriever.search(query, top_k=3)
    """

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_dim = 384   # all-MiniLM-L6-v2 output dimension

    def build(self, chunks: List[Dict]) -> None:
        """
        Build the FAISS index from a list of chunk dicts.
        
        Args:
            chunks: list of chunk dicts from loader.py
        """
        print(f"🔨 Building vector index for {len(chunks)} chunks...")

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        self.embedding_dim = embeddings.shape[1]
        self.chunks = chunks

        # Inner product index (works like cosine similarity since embeddings are normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))

        print(f"✅ Index built: {self.index.ntotal} vectors, dim={self.embedding_dim}")

    def save(self, path: str = None) -> None:
        """Save index and chunk metadata to disk."""
        if path is None:
            path = config.VECTORSTORE_FILE

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Vector store saved to {path}")

    def load(self, path: str = None) -> bool:
        """
        Load index and chunk metadata from disk.
        Returns True if successful, False if not found.
        """
        if path is None:
            path = config.VECTORSTORE_FILE

        faiss_path = f"{path}.faiss"
        chunks_path = f"{path}_chunks.json"

        if not Path(faiss_path).exists():
            return False

        self.index = faiss.read_index(faiss_path)

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

        return True

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        Args:
            query: user's question
            top_k: number of results to return
        
        Returns:
            list of result dicts with similarity scores
        """
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        query_emb = embed_query(query).astype(np.float32)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        # Sort by similarity score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results
