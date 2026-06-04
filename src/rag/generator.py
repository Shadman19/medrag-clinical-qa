"""
src/rag/generator.py
---------------------
Generates grounded answers from retrieved context.
Uses a local HuggingFace model — no API key needed.
"""

import re
from typing import List, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

_pipeline = None


def get_pipeline():
    """Load generation pipeline once (singleton)."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text2text-generation",
            model=config.GENERATION_MODEL,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )
    return _pipeline


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build a RAG prompt from the query and retrieved chunks.
    
    Args:
        query: user question
        chunks: retrieved chunks from retriever
    
    Returns:
        formatted prompt string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['doc_title']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the following medical question based ONLY on the provided context.
Be concise, accurate, and cite which source supports your answer.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def generate_answer(query: str, chunks: List[Dict]) -> Dict:
    """
    Generate an answer given a query and retrieved context chunks.
    
    Returns dict with:
        - answer: generated text
        - prompt: the full prompt used
        - sources: list of source titles cited
        - model: model name used
    """
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the medical knowledge base to answer this question.",
            "prompt": "",
            "sources": [],
            "model": config.GENERATION_MODEL,
        }

    prompt = build_prompt(query, chunks)

    try:
        pipe = get_pipeline()
        outputs = pipe(prompt, max_new_tokens=config.MAX_NEW_TOKENS)
        answer_text = outputs[0]["generated_text"].strip()
    except Exception as e:
        # Fallback: extractive answer from top chunk
        answer_text = _extractive_fallback(query, chunks)

    sources = list({c["doc_title"] for c in chunks})

    return {
        "answer": answer_text,
        "prompt": prompt,
        "sources": sources,
        "model": config.GENERATION_MODEL,
    }


def _extractive_fallback(query: str, chunks: List[Dict]) -> str:
    """
    Simple extractive fallback — returns the most relevant sentence
    from the top chunk when model inference fails.
    """
    if not chunks:
        return "No relevant information found."

    top_chunk = chunks[0]["text"]
    sentences = [s.strip() for s in re.split(r'[.!?]', top_chunk) if len(s.strip()) > 30]

    query_words = set(query.lower().split())
    best_sentence = ""
    best_overlap = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    if best_sentence:
        return f"{best_sentence}. (Source: {chunks[0]['doc_title']})"
    return sentences[0] if sentences else top_chunk[:300]
