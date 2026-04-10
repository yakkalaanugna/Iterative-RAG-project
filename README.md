# Automated Root Cause Analysis using RAG + Agent

An AI-powered **telecom automated testing log analyzer** that combines Retrieval-Augmented Generation (RAG) with an LLM Agent (Groq API) to automatically investigate logs, trace errors, and identify root causes.

Designed specifically for **telecom test execution environments** — supports eGate console logs, e2e test output, worker logs, syslog, and sosreport archives. Drop your test logs in, press Enter, and let the agent diagnose failures automatically.

## Who Is This For?

- Telecom engineers debugging **automated test failures**
- Teams working with **eGate simulator**, **e2e test frameworks**, or **Nokia RAN test environments**
- Anyone dealing with large, noisy test execution logs who wants AI-driven root cause analysis

## Features

- **Web Dashboard** — Streamlit-based UI for easy file upload and interactive analysis
- **Pass vs Fail Log Comparison** — upload both pass and fail logs, AI identifies errors unique to the fail run (ignoring common noise)
- **Auto-detect log format** — JSON, CSV, XML, HTML, plain text — parsed automatically
- **Telecom-specific prompt engineering** — LLM prompts tailored for eGate, e2e, Robot Framework, syslog
- Fast semantic search with FAISS and sentence-transformers
- Groq LLM Agent (Llama-3.3-70b-versatile) — free tier, no credit card needed
- **LLM-driven investigation** — agent decides which log files to check next
- Supports telecom log formats: eGate console, e2e output, worker logs, syslog, sosreport
- Auto-extracts archives (.tgz, .tar.gz, .zip) — no manual extraction needed
- Smart filtering — strips ANSI codes, deduplicates, keeps only important lines
- Auto-analyze mode (CLI) — just press Enter, no need to type a question
- Batch analysis — upload multiple files at once
- Secure API key management with `.env`

## Project Structure

```
├── app.py                # Web UI (Streamlit dashboard)
├── engine.py             # Core analysis engine (shared logic)
├── rag_system.py         # CLI analyzer
├── build_vector_store.py # Force-rebuild the vector store
├── retriever.py          # Simple query tool
├── vector_store.py       # Vector store builder
├── loader.py             # Log file loader
├── embedder.py           # Embedding helper
├── auto_rebuild.py       # Watch folder for changes
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
└── data/logs/            # Put your log files here
```

## Getting Started

**1. Clone the repo:**
```
git clone https://github.com/yakkalaanugna/Automated-RAG-Agent.git
cd Automated-RAG-Agent
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
Supported: `.log`, `.txt`, `.json`, `.csv`, `.xml`, `.html`, `.cfg`, `.tgz`, `.tar.gz`, `.zip`

**5. Run the Web UI:**
```
streamlit run app.py
```
This opens a browser dashboard where you can upload files, compare pass/fail logs, and get AI analysis.

**Or run the CLI analyzer:**
```
python rag_system.py
```
Press **Enter** to auto-analyze all logs, or type a specific question.

## Security

- **Never commit your `.env` file** — it is git-ignored by default.
- Only `.env.example` is included in the repo as a template.
