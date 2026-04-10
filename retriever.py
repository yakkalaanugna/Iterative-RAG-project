import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

VECTOR_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
LOG_FILE = os.path.join(VECTOR_DIR, "logs.pkl")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load saved vector store
index = faiss.read_index(INDEX_FILE)
with open(LOG_FILE, "rb") as f:
    logs = pickle.load(f)

print(f"System ready. {len(logs)} logs loaded.")

while True:
    query = input("\nEnter your question (type 'exit' to stop): ")

    # Stop condition
    if query.lower() == "exit":
        print("Stopping system...")
        break

    # Convert question to vector
    query_vector = model.encode([query])

    # Search similar logs
    k = 1
    distances, indices = index.search(query_vector, k)

    filename, log_message = logs[indices[0][0]]
    print("\nMost relevant log:")
    print(f"File: {filename}")
    print(f"Log: {log_message}")