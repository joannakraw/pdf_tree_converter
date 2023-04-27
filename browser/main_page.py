import streamlit as st
import pandas as pd
from io import StringIO

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
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

output_format = st.selectbox('Write tree to a format', ["newick", "phylo", "other"])