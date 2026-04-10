import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

LOG_FOLDER = "data/logs"
VECTOR_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
LOG_FILE = os.path.join(VECTOR_DIR, "logs.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

# Load logs from files
logs = []
for file in sorted(os.listdir(LOG_FOLDER)):
    if file.endswith(".txt"):
        path = os.path.join(LOG_FOLDER, file)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append((file, line))

if not logs:
    print("No log files found in", LOG_FOLDER)
    exit()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert logs to vectors
texts = [msg for _, msg in logs]
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save vector store
faiss.write_index(index, INDEX_FILE)
with open(LOG_FILE, "wb") as f:
    pickle.dump(logs, f)

print("Vector store created successfully")
print("Number of vectors stored:", index.ntotal)