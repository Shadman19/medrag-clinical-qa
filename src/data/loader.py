"""
src/data/loader.py
------------------
Loads medical documents and splits them into overlapping chunks
for indexing into the vector store.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def load_documents(documents_dir: str = None) -> List[Dict]:
    """
    Load all .txt documents from the documents directory.
    
    Returns:
        list of dicts with keys: id, title, source, content, filepath
    """
    if documents_dir is None:
        documents_dir = config.DOCUMENTS_DIR

    docs = []
    doc_dir = Path(documents_dir)

    if not doc_dir.exists():
        print(f"⚠️  Documents directory not found: {doc_dir}")
        return []

    for i, filepath in enumerate(sorted(doc_dir.glob("*.txt"))):
        content = filepath.read_text(encoding="utf-8")

        # Extract title and source from header lines
        title = filepath.stem.replace("_", " ").title()
        source = "Medical Reference"

        for line in content.split("\n")[:5]:
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()

        docs.append({
            "id": i,
            "title": title,
            "source": source,
            "content": content,
            "filepath": str(filepath),
            "filename": filepath.name,
        })

    return docs


def chunk_document(doc: Dict, chunk_size: int = None, overlap: int = None) -> List[Dict]:
    """
    Split a document into overlapping chunks.
    
    Strategy:
    - First try to split on section headers (ALL CAPS lines)
    - Then split long sections by character count with overlap
    
    Args:
        doc: document dict from load_documents()
        chunk_size: characters per chunk
        overlap: overlap characters between chunks
    
    Returns:
        list of chunk dicts
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP

    content = doc["content"]
    chunks = []

    # Split on section headers (lines in ALL CAPS)
    sections = re.split(r'\n(?=[A-Z][A-Z\s\/\(\)]+\n)', content)

    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Slide a window over the section
            start = 0
            while start < len(section):
                end = start + chunk_size
                chunk = section[start:end]
                if len(chunk.strip()) > 50:
                    chunks.append(chunk.strip())
                start += chunk_size - overlap

    # Build chunk dicts
    chunk_dicts = []
    for i, chunk_text in enumerate(chunks):
        chunk_dicts.append({
            "chunk_id": f"{doc['id']}_{i}",
            "doc_id": doc["id"],
            "doc_title": doc["title"],
            "doc_source": doc["source"],
            "doc_filename": doc["filename"],
            "chunk_index": i,
            "text": chunk_text,
        })

    return chunk_dicts


def load_and_chunk_all(documents_dir: str = None) -> List[Dict]:
    """
    Load all documents and return all chunks ready for embedding.
    """
    docs = load_documents(documents_dir)
    all_chunks = []

    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"   📄 {doc['title']}: {len(chunks)} chunks")

    return all_chunks
