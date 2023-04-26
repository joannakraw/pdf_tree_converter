# png_tree_converter

## Members
Aleksandra Cupriak, Agnieszka Kowalewska, Joana Krawczyk


## Project description
This is a final project for the Architecture of large projects in bioinformatics course. 
The goal of the project is motivated by the struggle to analyse, convert and download phylogenetic trees 
found in scientific articles. 

We propose Png Tree Converter - an online / command line (to be decided later)
tool converting phylogenetic trees in image format (.png) to a selected text file format (newick, phylo, nexus). 
Our approach is based on computer vision algorithms - edge detection with use of convolutional layer.
The tool can create visualisations of trees using BioPython and toytree libraries.

### Main tasks:
1. Extract dendrogram structure from .png image for binary, unrooted trees oriented vertically 
(having leaves at the bottom of the image)
2. Extract leaf names
3. Write dendrogram data to a tree file (ex. newick)

### Minor/extra tasks:
* Create an online tool using ```Streamlit```,
* adjust the algorithm for different tree orientations/types,
* create online tree visualisations (toytree, biopython etc.).

### Presentation from April 2023
[Link](https://docs.google.com/presentation/d/1AMbVaFBokSwe5lvQ4CdNTPvYihnjZUkXHr_3WURmY5s/edit?usp=sharing) to Google Slides.
