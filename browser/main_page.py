import importlib
import os.path
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import sys
sys.path.append('../')
import importlib
import TreeConverter as tc
importlib.reload(tc)

st.title(':deciduous_tree: Png Tree Converter')
st.sidebar.markdown("### Parameters")

st.write("This tool was created as a final project for ADP course at the University of Warsaw. "
         "It is designed to help analyse phylogenetic trees from publications that "
         "do not include trees in textual format such as .newick or .phylo. Here, we introduce a"
         " **Png Tree Converter** as an online tool for converting trees in .png format"
         "into a selected textual one. Try it yourself, upload a snapshot of a tree!")

fs = st.file_uploader("Choose a file with a tree image")
st.markdown("### :pencil2: Choose parameters")
left_column, middle_column, right_column = st.columns(3)
orientation = left_column.selectbox("Orientation", ['horizontal', 'vertical'])
min_freq = middle_column.number_input("Minimum frequency", min_value=0, value=100)
intersection_threshold = right_column.number_input("Intersection threshold", min_value=0, value=5)

resize_factor = st.slider("Resize factor", min_value=0.0, max_value=5.0, value=0.25, step=0.05)


if fs is not None:

    if not os.path.exists("temp/"):
        os.mkdir("temp/")
    image_path = "temp/image.png"

    with open(image_path, 'wb') as f:
        f.write(fs.read())

    st.write("Original image with phylogenetic tree")
    opencv_image = cv2.imread(image_path)
    st.image(opencv_image, channels="BGR")

    tc_image = tc.Image(image_path=image_path, resize_factor=resize_factor)
    st.markdown("#### Detected labels:")
    st.write(', '.join(tc_image.labels))
    # st.write(*tc_image.labels, sep=', ')
    st.pyplot(fig=tc_image.fig_boxes)
    st.image(cv2.imread("temp/image_tree.png"), caption="Cropped image after text removal")

    v_lines, h_lines, internal_nodes, leaves, fig_nodes_leaves = tc_image.tree_image.find_lines_intersections_leaves(legend=False,
                                                                                                   orientation=orientation,
                                                                                                   intersection_threshold=intersection_threshold,
                                                                                                   min_freq=min_freq)
    st.pyplot(fig=fig_nodes_leaves)

# output_format = st.selectbox('Write tree to a format', ["newick", "phylo", "other"])
# resize_factor = st.text_input("Choose resize parameter", 1)
# st.write(f"Selected parameters = {output_format, resize_factor}")