"""
src/evaluation/evaluate.py
---------------------------
Evaluates RAG pipeline quality using standard metrics:
- Retrieval precision: Are retrieved chunks relevant?
- Answer faithfulness: Does the answer stay grounded in context?
- Answer relevance: Does the answer address the question?
"""

import re
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Evaluation question set with expected keywords
EVAL_QUESTIONS = [
    {
        "question": "What are the symptoms of Type 2 diabetes?",
        "expected_keywords": ["polyuria", "polydipsia", "fatigue", "blurred", "thirst"],
        "expected_source": "Type 2 Diabetes",
    },
    {
        "question": "What is the first-line treatment for hypertension?",
        "expected_keywords": ["lifestyle", "diet", "exercise", "thiazide", "ACE"],
        "expected_source": "Hypertension",
    },
    {
        "question": "How is cardiovascular disease diagnosed?",
        "expected_keywords": ["ECG", "echocardiogram", "troponin", "angiography"],
        "expected_source": "Cardiovascular",
    },
    {
        "question": "What is the Gleason score in prostate cancer?",
        "expected_keywords": ["Gleason", "prostate", "grade", "aggressiveness"],
        "expected_source": "Cancer",
    },
    {
        "question": "What medications are used for depression?",
        "expected_keywords": ["SSRI", "sertraline", "fluoxetine", "antidepressant"],
        "expected_source": "Mental Health",
    },
    {
        "question": "What causes Type 2 diabetes?",
        "expected_keywords": ["insulin", "resistance", "beta cell", "pancreas"],
        "expected_source": "Type 2 Diabetes",
    },
    {
        "question": "What is HbA1c and what level indicates diabetes?",
        "expected_keywords": ["HbA1c", "6.5", "glycated", "hemoglobin"],
        "expected_source": "Type 2 Diabetes",
    },
    {
        "question": "What are the stages of hypertension?",
        "expected_keywords": ["Stage 1", "Stage 2", "130", "140", "normal"],
        "expected_source": "Hypertension",
    },
]


def compute_retrieval_precision(chunks: List[Dict], expected_keywords: List[str]) -> float:
    """
    Checks if retrieved chunks contain expected keywords.
    Returns fraction of expected keywords found in retrieved text.
    """
    if not chunks or not expected_keywords:
        return 0.0

    combined_text = " ".join(c["text"].lower() for c in chunks)
    found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return found / len(expected_keywords)


def compute_answer_faithfulness(answer: str, chunks: List[Dict]) -> float:
    """
    Measures whether the answer is grounded in retrieved context.
    Simple heuristic: fraction of answer sentences that overlap with context.
    """
    if not answer or not chunks:
        return 0.0

    context_text = " ".join(c["text"].lower() for c in chunks)
    answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]

    if not answer_sentences:
        return 0.0

    faithful = 0
    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        # Check if at least 40% of non-stopwords overlap with context
        content_words = {w for w in words if len(w) > 3}
        if content_words:
            overlap = sum(1 for w in content_words if w in context_text)
            if overlap / len(content_words) >= 0.4:
                faithful += 1

    return faithful / len(answer_sentences)


def compute_answer_relevance(answer: str, question: str) -> float:
    """
    Measures whether the answer is relevant to the question.
    Simple word overlap heuristic.
    """
    if not answer or not question:
        return 0.0

    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    content_q_words = {w for w in q_words if len(w) > 3}

    if not content_q_words:
        return 0.5

    overlap = len(content_q_words & a_words)
    return min(1.0, overlap / len(content_q_words))


def run_evaluation(pipeline) -> Dict:
    """
    Run full evaluation suite on the pipeline.
    
    Returns:
        dict with per-question results and aggregate metrics
    """
    results = []

    for eval_item in EVAL_QUESTIONS:
        result = pipeline.query(eval_item["question"])

        precision = compute_retrieval_precision(
            result["chunks"], eval_item["expected_keywords"]
        )
        faithfulness = compute_answer_faithfulness(
            result["answer"], result["chunks"]
        )
        relevance = compute_answer_relevance(
            result["answer"], eval_item["question"]
        )

        results.append({
            "question": eval_item["question"],
            "expected_source": eval_item["expected_source"],
            "actual_sources": result["sources"],
            "retrieval_precision": round(precision * 100, 1),
            "answer_faithfulness": round(faithfulness * 100, 1),
            "answer_relevance": round(relevance * 100, 1),
            "latency_ms": result["latency_ms"],
            "top_score": result["retrieval_scores"][0] if result["retrieval_scores"] else 0,
        })

    # Aggregate
    avg_precision = np.mean([r["retrieval_precision"] for r in results])
    avg_faithfulness = np.mean([r["answer_faithfulness"] for r in results])
    avg_relevance = np.mean([r["answer_relevance"] for r in results])
    avg_latency = np.mean([r["latency_ms"] for r in results])

    return {
        "results": results,
        "summary": {
            "avg_retrieval_precision": round(avg_precision, 1),
            "avg_answer_faithfulness": round(avg_faithfulness, 1),
            "avg_answer_relevance": round(avg_relevance, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "num_questions": len(results),
        }
    }
