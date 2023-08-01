from pathlib import Path
from argparse import ArgumentParser
from wget import download


def create_dirs():
    if not Path("assets").exists():
        Path("assets").mkdir(exist_ok=True)
    if not Path("models").exists():
        Path("models").mkdir(exist_ok=True)

    return True


def download_files():
    if create_dirs():
        print("'assets' & 'models' directories have been created")

        asset_urls = [
            "https://search-engine-files.s3.us-east-2.amazonaws.com/assets/annoy_word_embedding_index.ann",
            "https://search-engine-files.s3.us-east-2.amazonaws.com/assets/idx2word.json",
            "https://search-engine-files.s3.us-east-2.amazonaws.com/assets/resnet50V2_image_embeds_index.ann",
            "https://search-engine-files.s3.us-east-2.amazonaws.com/assets/word2idx.json",
        ]
        model_url = "https://search-engine-files.s3.us-east-2.amazonaws.com/models/resnet50v2_model-30-0.80.hdf5"

        print("Downlaoding the assets now, please be patient, this may take sometime.")
        # Download all the assets
        for url in asset_urls:
            download(url=url, out=f"assets\{url.split('/')[-1]}")

        print(
            "Downloaded the assets they are present under 'assets' folder, now dowloading the model weights...."
        )
        download(url=model_url, out="models\resnet50v2_model-30-0.80.hdf5")

        print(
            "Model weights have been dowload and are present under 'models\resnet50v2_model-30-0.80.hdf5'"
        )


if __name__ == "__main__":
    create_dirs()
    download_files()
