import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from annoy import AnnoyIndex
import json
import tensorflow as tf
from typing import Union

import src.config as CFG


def create_word_index(file_path: str):
    """
    Function to create AnnoyIndex of glove word embeddings & index-to-word dict.
    Incase they are already present just load and return.

    Parameters
    ----------
    file_path: Path to the glove txt file

    Returns
    -------
    idx2word: dictionary mapping all indexes to words
    word_embedding_index: AnnoyIndex of glove word vectors

    """
    word_embedding_index = AnnoyIndex(f=300, metric="angular")  # Init the Annoy Index
    idx2word = {}  # Init the dict
    word2idx = {}
    if (
        CFG.GLOVE_WORD_VECS_INDEX.exists()
        and CFG.IDX2WORD_PATH.exists()
        and CFG.WORD2IDX_PATH.exists()
    ):  # If both files exist then load and return them
        word_embedding_index.load(CFG.GLOVE_WORD_VECS_INDEX.__str__())
        with open(CFG.IDX2WORD_PATH, "r") as f:
            idx2word = json.load(fp=f)
        with open(CFG.WORD2IDX_PATH, "r") as f:
            word2idx = json.load(fp=f)
    else:  # Else create the AnnoyIndex as well as the JSON dict and save them
        with open(file_path, encoding="utf-8", mode="r") as f:
            for idx, line in tqdm(
                enumerate(f.readlines()),
                total=400000,
                desc="Processing glove vectors: ",
            ):  # Loop over each line in the file
                word, coefs = line.split(
                    maxsplit=1
                )  # Split the word and the glove vectors.
                coefs = np.fromstring(
                    coefs, "f", sep=" "
                )  # Convert the glove vectors to numpy array
                # Add the vector to index
                word_embedding_index.add_item(i=idx, vector=coefs)
                # Map the index to word
                idx2word[idx] = word

            word2idx = {v: k for k, v in idx2word.items()}

            print("Building the index")
            word_embedding_index.build(20)

            print("Saving to disk")
            word_embedding_index.save(CFG.GLOVE_WORD_VECS_INDEX.__str__())

            with open(
                file=CFG.IDX2WORD_PATH, mode="w"
            ) as f:  # Save the json dict as well
                json.dump(obj=idx2word, fp=f)

            with open(
                file=CFG.WORD2IDX_PATH, mode="w"
            ) as f:  # Save the reverse-json dict as well
                json.dump(obj=word2idx, fp=f)

    # Return both of them
    return idx2word, word_embedding_index, word2idx


def create_image_embedding_index(paths: list[str], model: tf.keras.Model):
    """
    Function to create AnnoyIndex of image embeddings & index-to-path dict.
    Incase they are already present just load and return.

    Parameters
    ----------
    paths: Path to all the images to index
    model: the DL model

    Returns
    -------
    idx2path: dictionary mapping all indexes to paths
    all_image_embeddings_indexed: AnnoyIndex of image embedding vectors

    """
    all_image_embedings_indexed = AnnoyIndex(
        f=300, metric="angular"
    )  # Init the AnnoyIndex

    idx2path = {
        k: v
        for k, v in tqdm(
            enumerate(CFG.TEST_PATHS),
            total=len(CFG.TEST_PATHS),
            desc="Creating index-to-path dict: ",
        )
    }  # Create the index to image path dict

    if CFG.IMAGE_FEATS_INDEX.exists():
        all_image_embedings_indexed.load(
            fn=CFG.IMAGE_FEATS_INDEX.__str__()
        )  # If AnnoyInedx exists then just load them
    else:
        for idx, pth in tqdm(
            enumerate(paths), total=len(paths), desc="Indexing images: "
        ):  # Loop over all the image paths
            image = (
                Image.open(pth).convert("RGB").resize((224, 224))
            )  # Open, convert, resize
            image = np.array(image) / 255.0  # Convert to numpy array and normalize

            embeds = model(image[np.newaxis, :])  # Get the embedding from the model

            all_image_embedings_indexed.add_item(
                idx, embeds.numpy().flatten()
            )  # Add it to the AnnoyIndex

        # Build the Index
        print("Building the index")
        all_image_embedings_indexed.build(n_trees=20)

        print("Saving the index to disk...")
        all_image_embedings_indexed.save(fn=CFG.IMAGE_FEATS_INDEX.__str__())

    # Return them
    return idx2path, all_image_embedings_indexed


def find_similar_tags(file: str, model: tf.keras.Model, num_tags: int):
    """
    Function to find tags for an image.

    Parameters
    ----------
    file: The image file
    model: The DL model
    num_tags: Number of tags to create

    Returns
    -------
    tags: Tags for the image
    """
    image = Image.open(file).convert("RGB").resize((224, 224))  # Open, convert, resize
    image = np.array(image) / 255.0  # Convert to numpy array & normalize

    embeds = model(image[np.newaxis, :])  # Get the image embedding from the model

    idx2word, annoy_word_embed_idx, _ = create_word_index(
        file_path=CFG.GLOVE_EMBEDDINGS
    )  # Get the idx2word dict & AnnoyIndex

    idxs = annoy_word_embed_idx.get_nns_by_vector(
        vector=embeds.numpy().flatten(), n=num_tags, search_k=-1
    )  # Search for the nearest `num_tags` vectors

    tags = [idx2word[str(idx)] for idx in idxs]  # Get the tags

    # Return the tags
    return tags


def find_similar_images(
    model: tf.keras.Model,
    num_images: int,
    search_by_word: Union[str, None] = None,
    image_pth: str = None,
):
    """
    Function to find similar for an image.

    Parameters
    ----------
    model: The DL model
    num_images: The number of images to recommend
    search_by_word: The word to search for similar images
    image_pth: Path to search image

    Returns
    -------
    paths: Paths to recommended images
    """
    idx2pth, ann_idx = create_image_embedding_index(paths=CFG.TEST_PATHS, model=model)

    if search_by_word is None:
        if image_pth is None:
            raise ValueError("An image needs to be provided")
        else:
            image = Image.open(image_pth).convert("RGB").resize((224, 224))
            image = np.array(image) / 255.0

            embeds = model(image[np.newaxis, :]).numpy().flatten()

            idxs = ann_idx.get_nns_by_vector(vector=embeds, n=num_images)

            paths = [idx2pth[idx] for idx in idxs]

            return paths
    else:
        _, word_ann, word2idx = create_word_index(
            file_path=CFG.GLOVE_EMBEDDINGS.__str__()
        )

        word_embed = word_ann.get_item_vector(word2idx[search_by_word])

        idxs = ann_idx.get_nns_by_vector(vector=word_embed, n=num_images)

        paths = [idx2pth[idx] for idx in idxs]

        return paths


def plot_images(paths: list[str], ncols: int, nrows: int):
    """
    Function to plot the images

    Parameters
    ----------
    paths: Paths to recommended images

    Returns
    -------
    fig: The matplotlib figure

    """
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            image = np.array(Image.open(paths[idx]).convert("RGB").resize((224, 224)))
            axs[i, j].imshow(image)
            axs[i, j].xaxis.set_visible(False)
            axs[i, j].yaxis.set_visible(False)
            idx += 1

    return fig
