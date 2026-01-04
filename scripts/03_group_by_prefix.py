import os
from collections import defaultdict

IMAGE_DIR = "../images"
OUTPUT_FILE = "../product_groups.txt"

groups = defaultdict(list)

for img in os.listdir(IMAGE_DIR):
    if "_" in img:
        prefix = img.split("_")[0]   # e.g., 1000 from 1000_031.jpg
        groups[prefix].append(img)

# Save groups
with open(OUTPUT_FILE, "w") as f:
    for prefix, imgs in groups.items():
        if len(imgs) > 1:  # only meaningful groups
            f.write(f"{prefix}: {', '.join(imgs)}\n")

print(f"Total product groups created: {len(groups)}")
print("Grouping saved to product_groups.txt")