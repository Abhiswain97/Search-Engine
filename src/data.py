import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

import config as CFG

from tqdm import tqdm
from PIL import Image


def load_glove_embeddings():
    embedding_dict = {}
    with open(file=CFG.GLOVE_EMBEDDINGS, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), total=400000):
            splits = line.split(maxsplit=1)
            embedding_dict[splits[0]] = np.fromstring(splits[1], dtype="f", sep=" ")

    np.save(file=CFG.GLOVE_EMBEDDINGS_INDEX, arr=embedding_dict, allow_pickle=True)


def create_labels2embeddings():
    em = np.load(CFG.GLOVE_EMBEDDINGS_INDEX, allow_pickle=True).tolist()

    final_embeddings = {}

    for label in np.unique(CFG.TRAIN_LABELS):
        sp_lbl = label.lower().split()
        if len(sp_lbl) > 1:
            final_embeddings[label] = (em[sp_lbl[0]] + em[sp_lbl[1]]) / 2
        else:
            final_embeddings[label] = em[label]

    np.save(
        file=CFG.LABEL_EMBEDDINGS_PATH,
        arr=np.array(final_embeddings),
        allow_pickle=True,
    )


# load_glove_embeddings()
# create_labels2embeddings()

# Create Dataframe
df = pd.DataFrame({"image_path": CFG.TRAIN_PATHS, "label": CFG.TRAIN_LABELS})

# Sample 1000 images from the DataFrame
df = (
    df.groupby(by="label")
    .apply(lambda x: x.sample(n=100, replace=True))
    .reset_index(drop=True)
)

# Split the dataset

train_df, val_df = train_test_split(df, test_size=0.20, stratify=df["label"].values)


# Creating the tf.data.Dataset and save to disk
def create_tf_dataset(df, mode, save=True):
    final_embeddings = np.load(
        file=CFG.LABEL_EMBEDDINGS_PATH, allow_pickle=True
    ).tolist()

    images, labels = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Load and resize the image
        image = Image.open(row["image_path"]).convert("RGB").resize((224, 224))

        # Normalize the image
        image = np.array(image) / 255.0

        # Get the embeeding for the corresponding label
        label = final_embeddings[row["label"]]

        # Create a list of images and labels
        images.append(image)
        labels.append(label)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if save:
        tf.data.experimental.save(
            dataset=ds, path=(CFG.SAVE_DATA_DIR / mode).as_posix()
        )


# create_tf_dataset(train_df, mode="Training", save=True)
# create_tf_dataset(val_df, mode="Validation", save=True)
