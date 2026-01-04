import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Files
EMB_FILE = "../embeddings.npy"
NAME_FILE = "../image_names.txt"

# Load data
embeddings = np.load(EMB_FILE)

with open(NAME_FILE, "r") as f:
    image_names = [line.strip() for line in f.readlines()]

print("Embeddings loaded:", embeddings.shape)

def get_top_k_similar(query_index, k=5):
    query_emb = embeddings[query_index].reshape(1, -1)

    similarities = cosine_similarity(query_emb, embeddings)[0]

    top_k_idx = np.argsort(similarities)[::-1][1:k+1]

    results = [(image_names[i], similarities[i]) for i in top_k_idx]
    return results

# -------------------------
# DEMO
# -------------------------
query_index = image_names.index("1000_031.jpg")
results = get_top_k_similar(query_index, k=5)

print(f"\nQuery image: {image_names[query_index]}")
print("Top similar images:")

for name, score in results:
    print(f"{name}  | similarity = {score:.4f}")
