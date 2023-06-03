import importlib
import os.path
import streamlit as st
import cv2
import sys
sys.path.append('../')
import importlib
import TreeConverter as tc
import NewickGenerator as ng
importlib.reload(tc)
importlib.reload(ng)
import shutil
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(':deciduous_tree: Png Tree Converter')

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
    st.pyplot(fig=tc_image.fig_boxes)

    v_lines, h_lines, internal_nodes, leaves, fig_nodes_leaves = tc_image.tree_image.find_lines_intersections_leaves(legend=False,
                                                                                                   orientation=orientation,
                                                                                                   intersection_threshold=intersection_threshold,
                                                                                                   min_freq=min_freq)
    st.pyplot(fig=fig_nodes_leaves)
    newick = ng.generate_newick_str(leaves, internal_nodes, orientation=orientation, labels=tc_image.labels)

    st.markdown("### Result check")
    fig_newick = ng.draw_newick(newick=newick)
    st.pyplot(fig=fig_newick)

    st.warning("If the tree plotted above is not correct, try tuning the parameters at the top of this page.")
    st.markdown("### Download a tree in newick format")

    st.download_button("Download tree", newick, file_name="newick_tree.nwk")
    shutil.rmtree("temp/")

