# png_tree_converter

## Members
Aleksandra Cupriak, Agnieszka Kowalewska, Joanna Krawczyk


## Project description
This is a final project for the Architecture of Large Projects in Bioinformatics course. 
The goal of the project is motivated by the struggle to analyse, convert and download phylogenetic trees 
found in scientific articles. 

We propose Png Tree Converter - an online tool converting phylogenetic trees in image format (.png) to newick format. 
Our approach is based on custom image processing algorithms.

### Main tasks:
1. Extract dendrogram structure from .png image for binary, 
rooted trees oriented vertically (having leaves at the bottom of the image).
2. Extract leaf names.
3. Write dendrogram data to a newick file.

### Extra tasks:
* Create an online tool using ```Streamlit```,
* adjust the algorithm for different tree orientations,
* create online tree visualisations (toytree, biopython etc.).

### Update
We fulfilled all main tasks. The first task was also extended for trees
oriented horizontally.
We also succeeded in developing a Streamlit application working locally. 
However, we cannot deploy our application on Streamlit 
Community Cloud as proposed in extra tasks, since we use ```OpenCV```
library which works on local computer, 
but will fail as soon as we launch it on Streamlit Community Cloud
[(explanation)](https://discuss.streamlit.io/t/problem-error-import-cv2/26348/7).

### Pipeline description
![pipeline.png](pipeline.png)

### How to run Streamlit app locally?
All the environment requirements are gathered in 
```environment.yml``` file. In order to run a Streamlit application on your computer you should:
1. Open the ```browser/``` folder in your terminal.
2. Paste ```streamlit run main_page.py```.
3. Access the application via the localhost link printed in your terminal.

### Project presentation
[Link](https://docs.google.com/presentation/d/1AMbVaFBokSwe5lvQ4CdNTPvYihnjZUkXHr_3WURmY5s/edit?usp=sharing) to Google Slides.
