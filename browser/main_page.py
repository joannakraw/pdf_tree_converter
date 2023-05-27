import streamlit as st
import pandas as pd
import cv2
import numpy as np

st.title(':deciduous_tree: Png Tree Converter')
st.sidebar.success("Select a page")

st.write("This tool was created as a final project for ADP course at the University of Warsaw. "
         "It is designed to help with analysing phylogenetic trees from publications that "
         "do not include trees in tree format such as .newick of .phylo. Here, we introduce a"
         " **Png Tree Converter** as an online tool for uploading trees in image format and "
         "converting them to a selected tree format. You can also create some basic"
         " visualisations in the visualisation page")

uploaded_file = st.file_uploader("Choose a file with a tree image")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.write("Original image with phylogenetic tree")
    st.image(opencv_image, channels="BGR")


output_format = st.selectbox('Write tree to a format', ["newick", "phylo", "other"])
resize_factor = st.text_input("Choose resize parameter", 1)
st.write(f"Selected parameters = {output_format, resize_factor}")