"""Standalone script to force-rebuild the vector store from data/logs/."""
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_system import scan_log_folder

LOG_FOLDER = "data/logs"
VECTOR_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
LOG_FILE = os.path.join(VECTOR_DIR, "logs.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Scanning log folder...")
logs = scan_log_folder(LOG_FOLDER)

if not logs:
    print("No log files found in", LOG_FOLDER)
    exit()

print(f"Found {len(logs)} log entries from {len(set(f for f, _ in logs))} files.")

texts = [msg for _, msg in logs]
print(f"Encoding {len(texts)} entries...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

with open(LOG_FILE, "wb") as f:
    pickle.dump(logs, f)

print(f"Vector store built successfully: {index.ntotal} vectors saved.")
