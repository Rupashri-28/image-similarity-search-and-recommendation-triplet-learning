import tensorflow as tf
from keras import layers, Model
from keras.applications import ResNet50
from collections import defaultdict
import os

from triplet_dataset import create_triplets

IMAGE_DIR = "../images"
MODEL_SAVE_PATH = "../models/triplet_resnet_model.h5"
BATCH_SIZE = 16
EPOCHS = 10
MARGIN = 0.2

# -----------------------------
# Load product groups
# -----------------------------
groups = defaultdict(list)

for img in os.listdir(IMAGE_DIR):
    if "_" in img:
        prefix = img.split("_")[0]
        groups[prefix].append(img)

groups = {k: v for k, v in groups.items() if len(v) > 1}

# -----------------------------
# Triplet loss
# -----------------------------
def triplet_loss(_, y_pred):
    anchor, positive, negative = y_pred
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + MARGIN, 0.0))

# -----------------------------
# Base embedding model
# -----------------------------
base_cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_cnn.trainable = False  # Phase 1

x = base_cnn.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

embedding_model = Model(base_cnn.input, x)

# -----------------------------
# Triplet model
# -----------------------------
anchor_in = layers.Input((224, 224, 3))
pos_in = layers.Input((224, 224, 3))
neg_in = layers.Input((224, 224, 3))

anchor_emb = embedding_model(anchor_in)
pos_emb = embedding_model(pos_in)
neg_emb = embedding_model(neg_in)

triplet_model = Model(
    inputs=[anchor_in, pos_in, neg_in],
    outputs=[anchor_emb, pos_emb, neg_emb]
)

triplet_model.add_loss(triplet_loss(None, triplet_model.output))
triplet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# -----------------------------
# Train
# -----------------------------
train_gen = create_triplets(groups, batch_size=BATCH_SIZE)

triplet_model.fit(train_gen, steps_per_epoch=100, epochs=EPOCHS)

# -----------------------------
# Fine-tuning (Phase 2)
# -----------------------------
for layer in base_cnn.layers[-30:]:
    layer.trainable = True

triplet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4)
)

triplet_model.fit(train_gen, steps_per_epoch=100, epochs=5)

# -----------------------------
# Save embedding model
# -----------------------------
embedding_model.save(MODEL_SAVE_PATH)
print("Triplet model training complete. Model saved.")
