# Search-Engine

### The project has an accompanying blog: [Building an Image Search/Tagging Engine](https://medium.com/@abhi08as-as/building-an-image-search-tagging-engine-285509b88c63)

Search or tag images

![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/b660fe63-faaf-4f45-afe0-f2082bc14cb1)

![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/91baa77c-ed1c-4db0-9215-5918e416de4c)

![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/910039de-bd9e-4189-9636-fc3b35b10be7)

![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/0ac91139-aa91-4cd1-8914-092533a04a52)

## How to run

1. Clone the repository: `git clone https://github.com/Abhiswain97/Search-Engine.git`
2. Install the requirements: `pip install -r requirements.txt`
3. Downloading the imagenette2-320 file:
   - Download the folder from this [link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)
   - This will be a ".tgz" file. Extract it.
   - You will get a ".tar" file inside it, extract that as well.
   - Now you need to just copy the "val" folder to the first "imagenette2-320" folder, rest you can delete. It should look something like this:
  
     ![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/df6c0e39-a007-4f9e-aae5-05c4bbefd6c5)
4. Next, run `python downloader.py` to download and setup all the folders and files. This step will take some time.
5. You final folder structure should look something like this:

   ![image](https://github.com/Abhiswain97/Search-Engine/assets/54038552/dc6b1b7d-2ff1-43b5-80cb-3d2ac71fa7fa)

6. You are now ready to run the app! Just do `streamlit run streamlit_app.py`

## References

1. [Building an image search service from scratch](https://blog.insightdatascience.com/the-unreasonable-effectiveness-of-deep-learning-representations-4ce83fc663cf)
2. [semantic-search](https://github.com/hundredblocks/semantic-search)
