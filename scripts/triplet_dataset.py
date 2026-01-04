import os
import random
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

IMAGE_DIR = "../images"
IMG_SIZE = (224, 224)

def load_image(img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img

def create_triplets(product_groups, batch_size=32):
    prefixes = list(product_groups.keys())

    # Sort prefixes by group size (larger groups = harder negatives)
    prefixes_sorted = sorted(
        prefixes,
        key=lambda p: len(product_groups[p]),
        reverse=True
    )

    while True:
        anchors, positives, negatives = [], [], []

        for _ in range(batch_size):
            # Positive class
            pos_prefix = random.choice(prefixes)

            # Semi-hard negative: choose from top frequent classes
            candidate_negatives = [
                p for p in prefixes_sorted[:len(prefixes_sorted)//2]
                if p != pos_prefix
            ]

            neg_prefix = random.choice(candidate_negatives)

            anchor, positive = random.sample(product_groups[pos_prefix], 2)
            negative = random.choice(product_groups[neg_prefix])

            anchors.append(load_image(anchor))
            positives.append(load_image(positive))
            negatives.append(load_image(negative))

        yield (
            [np.array(anchors), np.array(positives), np.array(negatives)],
            np.zeros(batch_size)
        )
