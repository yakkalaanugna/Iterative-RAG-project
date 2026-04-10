import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Initialize Groq client
# -----------------------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not set. Create a .env file with your key.")
    print("  Example: GROQ_API_KEY=gsk_your_key_here")
    exit(1)

groq_client = Groq(api_key=api_key)

# -----------------------------
# External AI Analysis (Groq)
# -----------------------------
def analyze_with_llm(log_text):
    """
    Send log to Groq API for analysis
    """

    if groq_client is None:
        return "AI Error: No API key configured. Set GROQ_API_KEY."

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a telecom log analysis assistant."
                },
                {
                    "role": "user",
                    "content": f"Analyze this log and give root cause and possible reason:\n{log_text}"
                }
            ],
            max_tokens=150
        )

        return response.choices[0].message.content

    except Exception as e:
        print("API Error:", e)
        return f"AI Error: {e}"


# -----------------------------
# Detect severity
# -----------------------------
def detect_severity(log_message):

    log_lower = log_message.lower()

    if "error" in log_lower:
        return "ERROR"

    elif "data loss" in log_lower:
        return "ERROR"

    elif "timeout" in log_lower:
        return "ERROR"

    elif "failed" in log_lower:
        return "FAIL"

    elif "warning" in log_lower:
        return "WARNING"

    elif "latency" in log_lower:
        return "WARNING"

    elif "delay" in log_lower:
        return "WARNING"

    elif "congestion" in log_lower:
        return "WARNING"

    else:
        return "INFO"


# -----------------------------
# Log cleaning & filtering
# -----------------------------
import re
import tarfile
import zipfile

def clean_line(line):
    """Strip ANSI escape codes and timestamps."""
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)  # remove ANSI codes
    line = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d+ \d{2}:\d{2}:\d{2}\.\d+ ', '', line)  # remove timestamps
    return line.strip()

IMPORTANT_KEYWORDS = [
    "warn", "error", "err", "fail", "timeout", "loss",
    "latency", "delay", "congestion", "critical", "fatal",
    "refused", "rejected", "denied", "abort", "crash",
    "exception", "unreachable", "invalid", "mismatch",
    "obsolete", "drop", "retry", "disconnect",
    "panic", "oom", "killed", "segfault", "oops",
    "not found", "degraded", "down", "offline"
]

def is_important(line):
    """Check if a log line contains something worth analyzing."""
    lower = line.lower()
    return any(kw in lower for kw in IMPORTANT_KEYWORDS)

def read_filtered_logs(filepath, filename):
    """Read a log file, filter to important unique lines."""
    seen = set()
    results = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cleaned = clean_line(line)
            if not cleaned or len(cleaned) < 10:
                continue
            if is_important(cleaned) and cleaned not in seen:
                seen.add(cleaned)
                results.append((filename, cleaned))
    return results

SUPPORTED_EXTENSIONS = (".txt", ".log", ".json", ".cfg")

# Files to look for inside sosreport/syslog archives
ARCHIVE_IMPORTANT_PATTERNS = [
    "syslog", "messages", "dmesg", "journalctl", "kern.log",
    "daemon.log", "boot.log", "auth.log", "cron.log",
    "worker", "egate", "alarm", "error"
]

def extract_and_read_archive(archive_path, archive_name):
    """Extract important log files from .tgz/.zip archives and filter them."""
    results = []
    print(f"  Extracting archive: {archive_name}...")

    try:
        if archive_path.endswith((".tgz", ".tar.gz")):
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    member_lower = member.name.lower()
                    # Only read text-like log files that match important patterns
                    if not any(p in member_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    if member.size > 50 * 1024 * 1024:  # skip files > 50MB
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        content = f.read().decode("utf-8", errors="ignore")
                        seen = set()
                        short_name = f"{archive_name}/{os.path.basename(member.name)}"
                        for line in content.splitlines():
                            cleaned = clean_line(line)
                            if not cleaned or len(cleaned) < 10:
                                continue
                            if is_important(cleaned) and cleaned not in seen:
                                seen.add(cleaned)
                                results.append((short_name, cleaned))
                    except Exception:
                        continue

        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for name in zf.namelist():
                    name_lower = name.lower()
                    if not any(p in name_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    info = zf.getinfo(name)
                    if info.file_size > 50 * 1024 * 1024:
                        continue
                    try:
                        content = zf.read(name).decode("utf-8", errors="ignore")
                        seen = set()
                        short_name = f"{archive_name}/{os.path.basename(name)}"
                        for line in content.splitlines():
                            cleaned = clean_line(line)
                            if not cleaned or len(cleaned) < 10:
                                continue
                            if is_important(cleaned) and cleaned not in seen:
                                seen.add(cleaned)
                                results.append((short_name, cleaned))
                    except Exception:
                        continue

    except Exception as e:
        print(f"  Warning: Could not extract {archive_name}: {e}")

    print(f"  Extracted {len(results)} important entries from {archive_name}")
    return results

ARCHIVE_EXTENSIONS = (".tgz", ".tar.gz", ".zip")

def scan_log_folder(folder):
    """Scan folder recursively for log files and archives."""
    all_logs = []

    for root, dirs, files in os.walk(folder):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, folder)

            if file.endswith(SUPPORTED_EXTENSIONS):
                entries = read_filtered_logs(filepath, rel_path)
                if entries:
                    print(f"  {rel_path}: {len(entries)} entries")
                all_logs.extend(entries)

            elif file.endswith(ARCHIVE_EXTENSIONS):
                entries = extract_and_read_archive(filepath, rel_path)
                all_logs.extend(entries)

    return all_logs


# -----------------------------
# Paths
# -----------------------------
VECTOR_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
LOG_FILE = os.path.join(VECTOR_DIR, "logs.pkl")
LOG_FOLDER = "data/logs"

os.makedirs(VECTOR_DIR, exist_ok=True)


# -----------------------------
# Initial check files (scanned first)
# -----------------------------
INITIAL_CHECK_PATTERNS = ["egate_console.log", "e2e_console_output.log", "log.html"]


# -----------------------------
# LLM functions
# -----------------------------
def ask_llm(messages):
    """Send messages to Groq LLM and get response."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API Error:", e)
        return f"AI Error: {e}"


def get_initial_errors(logs):
    """Get errors from initial check files (egate, e2e, log.html)."""
    initial_errors = {}
    for filename, msg in logs:
        fname_lower = filename.lower()
        for pattern in INITIAL_CHECK_PATTERNS:
            if pattern in fname_lower or pattern.replace(".", "") in fname_lower.replace(".", ""):
                if filename not in initial_errors:
                    initial_errors[filename] = []
                initial_errors[filename].append(msg)
                break
    return initial_errors


def get_available_files(logs):
    """Get list of unique file names in the vector store."""
    return sorted(set(filename for filename, _ in logs))


def get_logs_from_file(logs, file_pattern):
    """Get all log entries matching a file pattern."""
    pattern_lower = file_pattern.lower()
    return [(f, m) for f, m in logs if pattern_lower in f.lower()]


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Auto-detect and embed new logs
# -----------------------------
def load_new_logs(model, index, logs):
    """Check for new log files/archives and add them incrementally."""

    existing_files = set(filename for filename, _ in logs)
    new_logs = []

    all_new = scan_log_folder(LOG_FOLDER)
    for filename, log_message in all_new:
        if filename not in existing_files:
            new_logs.append((filename, log_message))

    existing_messages = set(msg for _, msg in logs)
    new_logs = [(f, m) for f, m in new_logs if m not in existing_messages]

    if new_logs:
        print(f"\nNew logs detected: {len(new_logs)} entries from new files.")
        texts = [msg for _, msg in new_logs]
        print(f"Encoding {len(texts)} new entries...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        logs.extend(new_logs)

        faiss.write_index(index, INDEX_FILE)
        with open(LOG_FILE, "wb") as f:
            pickle.dump(logs, f)

        print(f"Vector store updated: {index.ntotal} total vectors.")
    else:
        print("No new log files detected.")

    return index, logs


# -----------------------------
# Load or build vector store (auto-rebuilds if log folder changed)
# -----------------------------
LOG_HASH_FILE = os.path.join(VECTOR_DIR, "log_hash.txt")

def get_log_folder_hash():
    """Get a simple hash of file names + sizes in the log folder."""
    entries = []
    for root, dirs, files in os.walk(LOG_FOLDER):
        for f in sorted(files):
            fp = os.path.join(root, f)
            entries.append(f"{os.path.relpath(fp, LOG_FOLDER)}:{os.path.getsize(fp)}")
    return "|".join(entries)

def needs_rebuild():
    """Check if vector store needs rebuilding."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(LOG_FILE):
        return True
    if not os.path.exists(LOG_HASH_FILE):
        return True
    with open(LOG_HASH_FILE, "r") as f:
        old_hash = f.read().strip()
    return old_hash != get_log_folder_hash()

if needs_rebuild():
    print("Log folder changed or no vector store. Building...")
    print("Scanning log folder (including archives and subdirectories)...")
    logs = scan_log_folder(LOG_FOLDER)

    if not logs:
        print("No log files found in", LOG_FOLDER)
        exit()

    texts = [msg for _, msg in logs]
    print(f"Encoding {len(texts)} entries (this may take a moment)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(LOG_FILE, "wb") as f:
        pickle.dump(logs, f)
    with open(LOG_HASH_FILE, "w") as f:
        f.write(get_log_folder_hash())

    print(f"Vector store built: {index.ntotal} vectors saved.")
else:
    print("Loading existing vector store...")
    index = faiss.read_index(INDEX_FILE)
    with open(LOG_FILE, "rb") as f:
        logs = pickle.load(f)
    print(f"Loaded {index.ntotal} vectors.")

print("AI Log Analyzer ready.")
available_files = get_available_files(logs)
print(f"Available log files: {len(available_files)}")
for f in available_files:
    count = sum(1 for fn, _ in logs if fn == f)
    print(f"  {f} ({count} entries)")


# -----------------------------
# MAIN LOOP — LLM-driven investigation
# -----------------------------
while True:

    query = input("\nAsk a question (or press Enter to auto-analyze, 'exit' to stop): ").strip()

    if query.lower() == "exit":
        print("Stopping system...")
        break

    # Default query if user just hits Enter
    if not query:
        query = "Analyze all errors and find root cause"

    # STEP 1: Initial scan — check egate_console.log, e2e_console_output.log, log.html
    print("\n--- Step 1: Initial scan (egate_console, e2e_console, log.html) ---")
    initial_errors = get_initial_errors(logs)

    error_summary = ""
    for fname, entries in initial_errors.items():
        error_summary += f"\n[{fname}] ({len(entries)} issues):\n"
        for entry in entries[:15]:  # limit per file to avoid token overflow
            error_summary += f"  - {entry}\n"

    if not error_summary:
        # Fallback: use FAISS to find relevant logs
        print("No initial check files found. Using vector search...")
        query_vector = model.encode([query])
        distances, indices = index.search(query_vector, min(5, len(logs)))
        error_summary = "\n[Vector search results]:\n"
        for i in range(min(5, len(logs))):
            fname, msg = logs[indices[0][i]]
            error_summary += f"  - [{fname}] {msg}\n"

    file_list = "\n".join(f"  - {f}" for f in available_files)

    # STEP 2: Send to LLM — ask what to check next
    print("\n--- Step 2: Asking AI for initial diagnosis ---")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a telecom log debugging assistant. "
                "The user will give you initial error logs and a list of available files. "
                "First give a brief initial diagnosis. "
                "Then tell which file(s) to check next for deeper debugging. "
                "Reply with EXACTLY this format at the end:\n"
                "CHECK_NEXT: filename1, filename2\n"
                "Or if no more files needed:\n"
                "CHECK_NEXT: DONE"
            )
        },
        {
            "role": "user",
            "content": (
                f"User question: {query}\n\n"
                f"Initial error scan:{error_summary}\n\n"
                f"Available files:\n{file_list}"
            )
        }
    ]

    response = ask_llm(messages)
    print("\nAI Initial Diagnosis:")
    print(response)

    messages.append({"role": "assistant", "content": response})

    # STEP 3: Iterative deep-dive — LLM asks for files, we provide them
    MAX_ITERATIONS = 3
    iteration = 0

    while iteration < MAX_ITERATIONS:
        # Parse CHECK_NEXT from response
        check_next = None
        for line in response.splitlines():
            if line.strip().upper().startswith("CHECK_NEXT:"):
                check_next = line.split(":", 1)[1].strip()
                break

        if not check_next or check_next.upper() == "DONE":
            print("\n--- AI investigation complete ---")
            break

        iteration += 1
        requested_files = [f.strip() for f in check_next.split(",")]
        print(f"\n--- Step {iteration + 2}: AI requested files: {requested_files} ---")

        # Gather logs from requested files
        deep_logs = ""
        for pattern in requested_files:
            matched = get_logs_from_file(logs, pattern)
            if matched:
                deep_logs += f"\n[{pattern}] ({len(matched)} entries):\n"
                for fname, msg in matched[:20]:  # limit to prevent token overflow
                    deep_logs += f"  - {msg}\n"
            else:
                deep_logs += f"\n[{pattern}]: No matching file found.\n"

        if not deep_logs.strip():
            print("No matching files found for LLM request.")
            break

        # Send deeper logs to LLM
        messages.append({
            "role": "user",
            "content": (
                f"Here are the logs from the files you requested:\n{deep_logs}\n\n"
                "Analyze these and give deeper root cause analysis. "
                "If you need more files, reply with CHECK_NEXT: filename1, filename2\n"
                "If investigation is complete, reply with CHECK_NEXT: DONE"
            )
        })

        response = ask_llm(messages)
        print("\nAI Deep Analysis:")
        print(response)

        messages.append({"role": "assistant", "content": response})

    print("\n" + "=" * 60)