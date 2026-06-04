# config.py — All settings in one place

# ── Embedding Model ──────────────────────────────────────────
# Free, runs locally, no API key needed
# Best balance of speed and accuracy for biomedical text
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval Settings ───────────────────────────────────────
TOP_K_RETRIEVAL = 3        # Number of chunks to retrieve per query
CHUNK_SIZE = 400           # Characters per chunk (not tokens)
CHUNK_OVERLAP = 80         # Overlap between consecutive chunks

# ── Generation Settings ──────────────────────────────────────
# Uses a local model — no OpenAI API key needed
# For better answers, change to: "google/flan-t5-large"
GENERATION_MODEL = "google/flan-t5-base"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3

# ── Paths ─────────────────────────────────────────────────────
DOCUMENTS_DIR = "data/documents"
VECTORSTORE_DIR = "data/vectorstore"
VECTORSTORE_FILE = "data/vectorstore/faiss_index"

# ── UI Settings ───────────────────────────────────────────────
APP_TITLE = "MedRAG — Clinical Question Answering"
APP_SUBTITLE = "Ask any medical question. Get grounded, source-cited answers."
MAX_HISTORY = 10           # Number of past Q&As to keep in session

# ── Evaluation ────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 20      # Questions to use for evaluation run
