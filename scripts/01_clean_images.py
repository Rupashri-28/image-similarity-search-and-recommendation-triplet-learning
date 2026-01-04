from PIL import Image
import os

IMAGE_DIR = "../images"

bad_images = []

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    try:
        with Image.open(img_path) as img:
            img.verify()   # checks if image is corrupted
    except Exception as e:
        bad_images.append(img_name)

print(f"Total bad images found: {len(bad_images)}")

for img in bad_images:
    os.remove(os.path.join(IMAGE_DIR, img))

print("Corrupted images removed.")
