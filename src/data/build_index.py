"""
src/data/build_index.py
------------------------
Loads documents, creates chunks, embeds them, and saves FAISS index.
Run this ONCE before launching the app.

Usage:
    python src/data/build_index.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import load_and_chunk_all
from src.rag.retriever import MedicalRetriever

def build():
    print("=" * 55)
    print("  MedRAG — Building Knowledge Base Index")
    print("=" * 55)
    print()

    print("📂 Loading and chunking documents...")
    chunks = load_and_chunk_all()
    print(f"\n   Total chunks: {len(chunks)}")

    print("\n🔢 Embedding chunks (downloading model on first run ~80MB)...")
    retriever = MedicalRetriever()
    retriever.build(chunks)
    retriever.save()

    print("\n✅ Done! Knowledge base is ready.")
    print("   Next: streamlit run app.py")

if __name__ == "__main__":
    build()
