import os
import random
from collections import defaultdict

IMAGE_DIR = "../images"
TRAIN_FILE = "../train.txt"
TEST_FILE = "../test.txt"

# Group images by prefix
groups = defaultdict(list)

for img in os.listdir(IMAGE_DIR):
    if "_" in img:
        prefix = img.split("_")[0]
        groups[prefix].append(img)

prefixes = list(groups.keys())
random.seed(42)
random.shuffle(prefixes)

split_ratio = 0.8
split_idx = int(len(prefixes) * split_ratio)

train_prefixes = prefixes[:split_idx]
test_prefixes = prefixes[split_idx:]

train_images = []
test_images = []

for p in train_prefixes:
    train_images.extend(groups[p])

for p in test_prefixes:
    test_images.extend(groups[p])

with open(TRAIN_FILE, "w") as f:
    for img in train_images:
        f.write(img + "\n")

with open(TEST_FILE, "w") as f:
    for img in test_images:
        f.write(img + "\n")

print("Dataset split completed (by product groups).")
print(f"Train images: {len(train_images)}")
print(f"Test images: {len(test_images)}")
