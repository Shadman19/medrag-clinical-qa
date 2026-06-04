# MedRAG — Complete Usage Guide
## From Zero to Running App + GitHub in 15 Minutes

---

## PART 1 — Run It Right Now (30 Seconds, No Setup)

The file `static/index.html` is a FULLY SELF-CONTAINED app.

1. Find the file `static/index.html` on your computer
2. Double-click it
3. It opens in your browser — fully working, no internet needed

That's it. You can already use it and share screenshots.

---

## PART 2 — Run the Full Python Backend (Better Results)

The Python backend uses real HuggingFace transformer models for much better answers.

### Step 1: Install Python
- Go to https://www.python.org/downloads/
- Download Python 3.10+, install it
- On Windows: check "Add Python to PATH" during install

### Step 2: Open Terminal
- Windows: press Win+R, type `cmd`, press Enter
- Mac: press Cmd+Space, type `terminal`, press Enter

### Step 3: Go to the project folder
```
cd path/to/medrag
```
(replace `path/to/medrag` with wherever you extracted the ZIP)

### Step 4: Create virtual environment
```
python -m venv venv
```
Then activate it:
```
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```
You'll see `(venv)` appear — that means it's active.

### Step 5: Install dependencies
```
pip install -r requirements.txt
```
This downloads everything needed (~500MB on first run, takes 5 min).

### Step 6: Build the knowledge base
```
python src/data/build_index.py
```
This downloads the AI embedding model and indexes all 55 medical passages.
Takes ~2 minutes on first run. Shows:
```
✅ Index built: 55 vectors
✅ Vector store saved
✅ Done! Knowledge base is ready.
```

### Step 7: Launch the app
```
streamlit run app.py
```
Your browser opens at http://localhost:8501

---

## PART 3 — Upload to GitHub

### Step 1: Go to GitHub
Open https://github.com and sign in (or create account)

### Step 2: Create new repository
- Click "+" top right → "New repository"
- Name: `medrag-clinical-qa`
- Description: `Production RAG pipeline for medical question answering with FAISS vector store and interactive web UI`
- Set to Public
- Click "Create repository"

### Step 3: Upload files
- Click "uploading an existing file"
- Extract your ZIP, open the medrag folder
- Press Ctrl+A to select ALL files inside
- Drag everything into GitHub
- Write commit message: `Initial commit — MedRAG RAG pipeline`
- Click "Commit changes"

### Step 4: Add topics (makes you discoverable)
- On your repo page, click the gear ⚙️ next to "About"
- Add topics: `rag` `nlp` `llm` `healthcare-ai` `faiss` `python` `streamlit` `biomedical-nlp`
- Click Save changes

Your repo is now live at:
`https://github.com/Shadman19/medrag-clinical-qa`

---

## PART 4 — Deploy Live Demo (FREE, Anyone Can Use It)

Deploy to Streamlit Cloud so anyone in the world can use your app:

### Step 1: Go to https://share.streamlit.io
Sign in with your GitHub account

### Step 2: Click "New app"
- Repository: `Shadman19/medrag-clinical-qa`
- Branch: `main`
- Main file path: `app.py`
- Click "Deploy"

Wait ~3 minutes. You get a live URL like:
`https://shadman19-medrag-clinical-qa.streamlit.app`

Share THIS link on LinkedIn — anyone clicks it and uses your app instantly.

---

## PART 5 — LinkedIn Post (Copy & Paste)

Post this on LinkedIn, attach a screenshot of the running app:

---

I built a production RAG (Retrieval-Augmented Generation) system for medical question answering from scratch.

No LangChain. No black boxes. Built every component myself.

The result: grounded, source-cited medical answers in under 500ms.

Here's the architecture 👇

Query → Sentence Transformer embedding → FAISS vector search → Context ranking → Answer generation → Source citation

What I had to solve:
• Chunking clinical text without destroying medical context
• Choosing the right embedding model for biomedical language  
• Measuring answer faithfulness (not just relevance)
• Making it usable by anyone — not just ML engineers

Live demo: [YOUR STREAMLIT URL]
GitHub: [YOUR GITHUB LINK — put in first comment]

#MachineLearning #RAG #NLP #HealthcareAI #LLM #OpenSource #Python

---

## Questions & Answers

**Q: Can I add more medical topics?**
Add a new .txt file to `data/documents/` with the same TITLE/SOURCE header format, then re-run `python src/data/build_index.py`.

**Q: Can I use a better model?**
In `config.py`, change `GENERATION_MODEL = "google/flan-t5-large"` for better answers.

**Q: What if I get an error?**
Most errors are solved by: 1) making sure your venv is activated, 2) re-running pip install -r requirements.txt

**Q: How do I explain this in an interview?**
"I built a RAG pipeline from scratch. The core challenge was building a retrieval system that finds semantically relevant medical passages — not just keyword matches — and then generating answers that are grounded in those passages rather than hallucinated. I also built an evaluation suite that measures retrieval precision and answer faithfulness."
