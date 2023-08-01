# Search-Engine
Search or tag images

## How to run

1. Install the requirements: `pip install -r requirements.txt`
2. Downloading the imagenette2-320 file:
   - Download the folder from this [link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)
   - This will be a ".tgz" file. Extract it.
   - You will get a ".tar" file inside it, extract that as well.
   - Now you need to just copy the "val" folder to the first "imagenette2-320" folder, rest you can delete. It should look something like this:
  
     ![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/df6c0e39-a007-4f9e-aae5-05c4bbefd6c5)
3. Next, run `python downloader.py` to download and setup all the folders and files. This step will take some time.
4. You final folder structure should look something like this:

   ![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/dc6b1b7d-2ff1-43b5-80cb-3d2ac71fa7fa)

5. You are now ready to run the app! Just do `streamlit run streamlit_app.py`
