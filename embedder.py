from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_embeddings(texts):

    embeddings = model.encode(texts)

    return embeddings


if __name__ == "__main__":

    from loader import load_logs

    logs = load_logs()

    vectors = create_embeddings(logs)

    print("Number of logs:", len(logs))

    print("Vector dimension:", len(vectors[0]))

    print("\nFirst vector sample:")

    print(vectors[0][:10])