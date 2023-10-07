import streamlit as st
from time import time
import tensorflow as tf

from src.utils import find_similar_images, find_similar_tags, plot_images

st.set_option("deprecation.showPyplotGlobalUse", False)


radio = st.sidebar.radio(
    label="Find images or tags ?", options=["Find images", "Find tags"]
)

st.markdown(
    "<h1><i><center>Find Images/Tags</center></i><h1>", unsafe_allow_html=True,
)

model = tf.keras.models.load_model(r"models\resnet50v2_model-30-0.80.hdf5")

if radio == "Find images":
    search = st.sidebar.radio(
        label="Search by image/text ?", options=["Search by image", "Search by text"]
    )

    if search == "Search by image":
        file = st.file_uploader("Upload image!")

        if file is not None:
            st.image(file, use_column_width=True)
            button = st.button("Find images")

            if button:
                with st.spinner("Finding images....."):
                    start = time()
                    paths = find_similar_images(
                        model=model, num_images=9, search_by_word=None, image_pth=file
                    )
                    fig = plot_images(paths=paths, ncols=3, nrows=3)
                    end = time() - start
                    st.success(f"Predcition done in: {end} secs")

                    st.pyplot(fig, use_container_width=True)
    else:
        word = st.text_input(label="Enter the text to search")

        if word != "":
            button = st.button("Find images")

            if button:
                with st.spinner("Finding images....."):
                    start = time()
                    paths = find_similar_images(
                        model=model, num_images=9, search_by_word=word
                    )
                    fig = plot_images(paths=paths, ncols=3, nrows=3)
                    end = time() - start
                    st.success(f"Predcition done in: {end} secs")

                    st.pyplot(fig, use_container_width=True)

else:
    file = st.file_uploader(label="Upload an image")
    num_tags = st.number_input("Enter number of tags")

    if file:
        st.image(file, use_column_width=True)
        button = st.button("Find tags")

        if button:
            with st.spinner("Finding tags....."):
                start = time()
                tags = find_similar_tags(file=file, model=model, num_tags=int(num_tags))
                end = time() - start
                st.success(f"Predcition done in: {end} secs")

                st.write(tags)
