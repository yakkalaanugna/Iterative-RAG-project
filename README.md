# Automated Root Cause Analysis using RAG + Agent

An AI-powered telecom log analyzer that combines Retrieval-Augmented Generation (RAG) with an LLM Agent (Groq API) to automatically investigate logs, trace errors, and identify root causes. Drop your log files in, press Enter, and let the agent do the rest.

## Features

- Fast vector search with FAISS and sentence-transformers
- Groq LLM integration (Llama-3.3-70b-versatile) — free tier, no credit card needed
- Auto-detects new logs and archives (.tgz, .zip) without manual rebuild
- Smart filtering — strips ANSI codes, deduplicates, keeps only important lines
- LLM-driven investigation — AI decides which files to check next
- Auto-analyze mode — just press Enter, no need to type a question
- Secure API key management with `.env`

## Project Structure

```
iterative-rag-project/
├── rag_system.py        # Main analyzer (run this)
├── build_vector_store.py # Force-rebuild the vector store
├── retriever.py         # Simple query tool
├── vector_store.py      # Vector store builder
├── loader.py            # Log file loader
├── embedder.py          # Embedding helper
├── auto_rebuild.py      # Watch folder for changes
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
└── data/logs/           # Put your log files here
```

## Getting Started

**1. Clone the repo:**
```
git clone https://github.com/yakkalaanugna/Iterative-RAG-project.git
cd Iterative-RAG-project
```

**2. Create a virtual environment and install requirements:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set up your API key:**
- Copy `.env.example` to `.env`
- Get a free Groq API key from https://console.groq.com/keys
- Add it to your `.env` file:
  ```
  GROQ_API_KEY=gsk_your_key_here
  ```

**4. Add your log files to `data/logs/`**  
Supported: `.log`, `.txt`, `.json`, `.cfg`, `.tgz`, `.tar.gz`, `.zip`

**5. Run the analyzer:**
```
python rag_system.py
```
Press **Enter** to auto-analyze all logs, or type a specific question.

## Security

- **Never commit your `.env` file** — it is git-ignored by default.
- Only `.env.example` is included in the repo as a template.
