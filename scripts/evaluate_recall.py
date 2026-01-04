import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
embeddings = np.load("../embeddings.npy")

with open("../image_names.txt", "r") as f:
    image_names = [line.strip() for line in f.readlines()]

# Extract product prefix (before "_")
def get_prefix(img_name):
    return img_name.split("_")[0]

# Compute Recall@K
def compute_recall_at_k(K):
    success = 0
    total = len(image_names)

    for i, query_name in enumerate(image_names):
        query_emb = embeddings[i].reshape(1, -1)
        sims = cosine_similarity(query_emb, embeddings)[0]

        # Exclude self
        ranked_indices = np.argsort(sims)[::-1]
        ranked_indices = [idx for idx in ranked_indices if idx != i]

        top_k = ranked_indices[:K]
        query_prefix = get_prefix(query_name)

        found = False
        for idx in top_k:
            if get_prefix(image_names[idx]) == query_prefix:
                found = True
                break

        if found:
            success += 1

    return success / total


if __name__ == "__main__":
    for k in [1, 5, 10]:
        recall = compute_recall_at_k(k)
        print(f"Recall@{k}: {recall:.4f}")
