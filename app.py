import streamlit as st
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ===============================
# Demo Configuration
# ===============================

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "triplet_resnet_model.h5")
EMB_PATH = os.path.join(ARTIFACT_DIR, "embeddings.npy")
IMAGE_NAMES_PATH = "image_names.txt"

IMG_SIZE = (224, 224)
TOP_K = 5

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="AI Visual Search Demo", layout="wide")

st.title("🔍 AI-Powered Visual Similarity Search")
st.caption("Demo mode — uses pre-trained Triplet Network (no training required)")

# ===============================
# Load Artifacts
# ===============================

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)

    embeddings = np.load(EMB_PATH)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    with open(IMAGE_NAMES_PATH, "r") as f:
        image_names = [line.strip() for line in f.readlines()]

    return model, embeddings, image_names


model, embeddings, image_names = load_artifacts()

# ===============================
# Image Preprocessing
# ===============================

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# ===============================
# Similarity Search
# ===============================

def find_similar_images(query_img, top_k=5):
    query_emb = model.predict(query_img)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = [(image_names[i], sims[i]) for i in top_indices]
    return results


# ===============================
# UI Logic
# ===============================

uploaded_file = st.file_uploader(
    "Upload an image to find visually similar products",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Query Image")
        st.image(query_image, use_column_width=True)

    with st.spinner("Searching for similar images..."):
        processed_img = preprocess_image(query_image)
        results = find_similar_images(processed_img, TOP_K)

    with col2:
        st.subheader(f"Top {TOP_K} Similar Images")

        for name, score in results:
            st.write(f"**{name}** — Similarity: `{score:.4f}`")

    st.success("Similarity search complete.")

else:
    st.info("Please upload an image to start the demo.")

# ===============================
# Footer
# ===============================

st.markdown("---")
st.caption(
    "This demo uses precomputed embeddings and a trained Triplet Network. "
    "Training scripts are available in the repository for reproducibility."
)