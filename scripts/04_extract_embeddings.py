import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Paths
IMAGE_DIR = "../images"
TRAIN_FILE = "../train.txt" 
EMB_FILE = "../embeddings.npy"
NAME_FILE = "../image_names.txt"

# Load image list
with open(TRAIN_FILE, "r") as f:
    image_names = [line.strip() for line in f.readlines()]

# Load pretrained ResNet50 (NO classifier)
model = load_model("../models/triplet_resnet_model.h5", compile=False)

def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = model.predict(x, verbose=0)
    return emb.flatten()

embeddings = []
valid_names = []

for name in tqdm(image_names):
    path = os.path.join(IMAGE_DIR, name)
    try:
        emb = extract_embedding(path)
        embeddings.append(emb)
        valid_names.append(name)
    except:
        print(f"Skipped {name}")

embeddings = np.array(embeddings)

np.save(EMB_FILE, embeddings)

with open(NAME_FILE, "w") as f:
    for n in valid_names:
        f.write(n + "\n")

print("DONE")
print("Embeddings shape:", embeddings.shape)