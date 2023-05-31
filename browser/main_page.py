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
st.sidebar.success("Select a page")

st.write("This tool was created as a final project for ADP course at the University of Warsaw. "
         "It is designed to help with analysing phylogenetic trees from publications that "
         "do not include trees in tree format such as .newick of .phylo. Here, we introduce a"
         " **Png Tree Converter** as an online tool for uploading trees in image format and "
         "converting them to a selected tree format. You can also create some basic"
         " visualisations in the visualisation page")

fs = st.file_uploader("Choose a file with a tree image")
orientation = st.selectbox("Select an orientation", ['horizontal', 'vertical'])

if fs is not None:

    if not os.path.exists("temp/"):
        os.mkdir("temp/")
    image_path = "temp/image.png"

    with open(image_path, 'wb') as f:
        f.write(fs.read())

    st.write("Original image with phylogenetic tree")
    opencv_image = cv2.imread(image_path)
    st.image(opencv_image, channels="BGR")

    tc_image = tc.Image(image_path=image_path, resize_factor=0.25)
    st.write(f"Detected labels: {', '.join(tc_image.labels)}")
    # st.write(*tc_image.labels, sep=', ')
    st.pyplot(fig=tc_image.fig_boxes)
    st.image(cv2.imread("temp/image_tree.png"), caption="Cropped image after text removal")

    v_lines, h_lines, internal_nodes, leaves, fig_nodes_leaves = tc_image.tree_image.find_lines_intersections_leaves(legend=False,
                                                                                                   orientation=orientation,
                                                                                                   intersection_threshold=5,
                                                                                                   min_freq=100)
    st.pyplot(fig=fig_nodes_leaves)

# output_format = st.selectbox('Write tree to a format', ["newick", "phylo", "other"])
# resize_factor = st.text_input("Choose resize parameter", 1)
# st.write(f"Selected parameters = {output_format, resize_factor}")