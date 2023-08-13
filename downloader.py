from pathlib import Path
import gdown


def create_dirs():
    if not Path("assets").exists():
        Path("assets").mkdir(exist_ok=True)
    if not Path("models").exists():
        Path("models").mkdir(exist_ok=True)


def download_files():
    create_dirs()
    print("'assets' & 'models' directories have been created")

    asset_url = (
        "https://drive.google.com/drive/u/0/folders/1cBWJDINmH6N3MzOqDxmBzeAQW4oL3mlI"
    )
    model_url = (
        "https://drive.google.com/drive/u/0/folders/13ce_FgTkDkZ3gxbjLaCEmp4ulH5lOjNK"
    )

    print("Downlaoding the assets now, please be patient, this may take sometime.")

    # Download all the assets
    gdown.download_folder(url=asset_url, output="assets")
    print(
        "Downloaded the assets they are present under 'assets' folder, now dowloading the model weights...."
    )

    gdown.download_folder(url=model_url, output="models")
    print(
        "Model weights have been dowload and are present under 'models\resnet50v2_model-30-0.80.hdf5'"
    )


if __name__ == "__main__":
    download_files()
