from pathlib import Path

BASE_PATH_PARENTS = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH_PARENTS / "imagenette2-320"

TRAIN_PATHS = list(DATA_PATH.glob("train/*/*.*"))
TEST_PATHS = list(DATA_PATH.glob("val/*/*.*"))

TRAIN_LABELS = [pth.parent.name.lower() for pth in TRAIN_PATHS]
TEST_LABELS = [pth.parent.name.lower() for pth in TEST_PATHS]

SAVE_DATA_DIR = BASE_PATH_PARENTS / "data"

GLOVE_EMBEDDINGS = BASE_PATH_PARENTS / "glove" / "glove.6B.300D.txt"
# GLOVE_EMBEDDINGS_INDEX = BASE_PATH_PARENTS / "assets" / "embeddings_index.npy"
# LABEL_EMBEDDINGS_PATH = BASE_PATH_PARENTS / "assets" / "label2embeddings.npy"

IMAGE_FEATS_INDEX = BASE_PATH_PARENTS / "assets" / "resnet50V2_image_embeds_index.ann"
GLOVE_WORD_VECS_INDEX = BASE_PATH_PARENTS / "assets" / "annoy_word_embedding_index.ann"

IDX2WORD_PATH = BASE_PATH_PARENTS / "assets" / "idx2word.json"
WORD2IDX_PATH = BASE_PATH_PARENTS / "assets" / "word2idx.json"

MODEL_PATH = BASE_PATH_PARENTS / "models" / "resnet50v2_model-30-0.80.hdf5"

# Hyper parameters
BATCH_SIZE = 16
LR = 0.01
