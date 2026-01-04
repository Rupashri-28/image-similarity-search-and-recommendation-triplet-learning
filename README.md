# AI-Powered Visual Similarity Search (Triplet Learning)

This repository contains a **demo-ready image similarity search system**
built using **deep metric learning with Triplet Networks**.

The system learns visual embeddings such that **images of the same product
are close together in embedding space**, enabling accurate visual search
and recommendation.

This project supports **two modes**:
-  **Demo Mode** (instant usage, no training)
-  **Training Mode** (full reproducible pipeline)

---

## Problem Statement

Traditional text-based image search fails to capture **visual similarity**.
In domains such as e-commerce and fashion, users want to find products that
*look similar*, not just share keywords.

**Goal:**  
Automatically retrieve visually similar images from a large image database
using **image content alone**, without relying on manual labels or metadata.

---

## Demo Mode (Recommended)

This repository includes **pre-trained artifacts** so that the system can
be demonstrated **without retraining**.

### How to Run the Demo

pip install -r requirements.txt
python app.py

Then

- Upload an image
- View the Top-K visually similar images
- Similarity is computed using cosine similarity in embedding space

### Demo Artifacts

The following files are included for demonstration purposes:

artifacts/triplet_resnet_model.h5
→ Trained Triplet Network (ResNet50 backbone)

artifacts/embeddings.npy
→ Precomputed 256-D embeddings for all dataset images

These allow instant similarity search without running the training pipeline.

### Training Mode (Reproducible Pipeline)

The repository also contains the complete training and evaluation pipeline
used to generate the demo artifacts.

### Pipeline Overview

- Image cleaning (remove corrupted files)
- Dataset splitting (train / test)
- Product grouping using product IDs
- Triplet generation (anchor, positive, negative)
- Triplet network training
- Embedding extraction 
- Similarity search
- Recall@K evaluation
- Training Scripts

All pipeline steps are available in the scripts/ directory and can be run
in sequence to reproduce results.

### Model Architecture

Backbone: Pretrained ResNet50
Embedding dimension: 256
Loss function: Triplet Loss (margin-based)
Similarity metric: Cosine Similarity

The model is trained to minimize distance between images of the same product
while maximizing distance from different products.

### Dataset

The dataset consists of fashion product images where multiple images
belong to the same product ID.

Due to size constraints, the dataset is not included in this repository.

Expected filename format:

<product_id>_<view_id>.jpg

Example:

1000_031.jpg
1000_032.jpg

### Evaluation
Recall@K Results
Metric	Value
Recall@1	36.14%
Recall@5	57.39%
Recall@10	65.49%

Recall@K Definition:
A query is counted as correct if at least one image of the same product
appears in the Top-K retrieved results.

These results indicate that the learned embeddings successfully cluster visually similar products, with retrieval quality improving as K increases.

Note: Recall@K values are dataset-dependent and can be further improved with longer training, hard-negative mining, and larger batch sizes.

### Validation Strategy

Due to limited labeled data, explicit validation splits were not used.
Validation was approximated using:

Triplet loss convergence during training

Qualitative inspection of retrieved results

Quantitative Recall@K evaluation

### Project Structure
.
├── artifacts/                # Demo artifacts
│   ├── triplet_resnet_model.h5
│   └── embeddings.npy
│
├── scripts/                  # Full training pipeline
│   ├── 01_clean_images.py
│   ├── 02_split_dataset.py
│   ├── 03_group_by_prefix.py
│   ├── 04_extract_embeddings.py
│   ├── 05_similarity_search.py
│   ├── 07_train_triplet_model.py
│   ├── evaluate_recall.py
│   └── triplet_dataset.py
│
├── app.py                    # Demo entry point
├── requirements.txt
├── README.md
└── .gitignore

### Future Work

- Replace brute-force cosine similarity with FAISS for scalable retrieval
- Hard-negative mining during training
- Mean Average Precision (mAP) evaluation
- API deployment using FastAPI

### Tech Stack

Python
TensorFlow / Keras
ResNet50
NumPy
scikit-learn
Streamlit

### Conclusion

This project demonstrates an end-to-end AI-powered image similarity search system using Triplet Networks and deep metric learning.

By learning meaningful visual embeddings directly from images, the system enables accurate similarity-based retrieval without relying on manual labels or text metadata - making it suitable for real-world applications such as e-commerce visual search and recommendation systems.