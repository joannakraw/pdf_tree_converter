import streamlit as st

st.markdown("## Project description")
'''
This is a final project for the Architecture of Large Projects in Bioinformatics course. 
The goal of the project is motivated by the struggle to analyse, convert and download phylogenetic trees 
found in scientific articles. 

We propose Png Tree Converter - an online tool converting phylogenetic trees in image format (.png) to newick format. 
Our approach is based on custom image processing algorithms.
'''

st.markdown("## Manual")

st.markdown('#### Pipeline description')
"""
1. The image provided by the user is 
    * read using ```OpenCV``` library, 
    * scaled based on user-defined ```resize_factor``` parameter,
    * converted to black and white scale and saved as a binary matrix.
2. Labels of the tree's leaves are detected (text and bounding boxes) using ```easyocr``` library. 
3. Labels' bounding boxes are occluded.
4. Nodes and leaves are detected based on tree image without textual data, using our custom image processing methods. (See Branches and nodes detection.)
5. The spatial location of detected internal nodes and leaves implies the structure of the tree.
6. The infered tree is saved to textual format.
"""

st.markdown('#### Branches and nodes detection')
'''
The algorithm for finding branches and nodes consists of the consecutive steps.
1. Find non-zero pixels of the binary array.
2. Sort rows and columns by non-zero pixels count and select those with the count $\geq$ ```min_freq```.
3. For each selected row and column, find lines contained in it. Define as a line a set of consecutive non-zero pixels, with gaps no longer than ```max_gap```.
A line cannot be shorter than ```min_line_length```.
4. Find all intersections between horizontal and vertical lines. An intersection is a common point of two lines, after prolonging them by 2 pixels from each end.
A point lying closer than ```intersection_threshold``` to an endpoint of each of the two intersecting lines is not considered an intersection.
5. Filter intersections - find groups of intersections lying in proximity of 10 pixels from each other and replace them with one intersection with mean coordinates.
6. Depending on the image orientation, find line ends aligned on the right or bottom of the image and save them as leaf candidates.
7. Filter leaves (similar to p. 5.).
8. Return filtered intersections and leaves.
'''

st.markdown('#### Parameters description and tuning')
"""
* ```orientation``` - 'horizontal' or 'vertical' - orientation of the input tree:
    * horizontal if leaf labels are aligned vertically on the right hand side of the image,
    * vertical if leaf labels are aligned horizontally at the bottom of the image.
* ```minimum frequency``` - int, default to 100 - as mentioned in Branches and nodes detection section, we detect horizontal and vertical lines based on the number of non-zero pixels aligned horizontally/vertically in an array-representation of the image; for this purpose we consider only the rows/columns where the number of non-zero pixels is greater or equal ```minimum frequency```; if there are short lines in the input image that are not detected, try decreasing ```minimum frequency```, but this may increase the computation time.
* ```instersection threshold``` - int - while detecting internal nodes, we consider all intersections between horizontal and vertical lines; we need to filter out the corners (intersections close to end of both of lines);  if for one of the intersecting lines, the distances (in pixels) from both ends to the intersection are greater than  ```instersection threshold```, then we consider this intersection as an internal node candidate;  if some corners are detected as internal nodes, try increasing ```instersection threshold```.
* ```resize factor``` - float, default to 0.25 - scaling factor of the input image after text detection and removal, used for coping with varying resolution of input images;  increasing ```resize factor``` may thicken the detected lines, consider doing it if some lines are thin.
"""

st.markdown('#### Output description')
'''
In order to make the final result more explainable and enable easier parameter tuning, the tool outputs the following consecutive figures.
1. The original image, in black and white (binary) scale.
2. The binary image with labels' bounding boxes (red).
3. Tree with detected internal nodes (cyan) and leaves (magenta).
4. Result check - the infered tree along with the labels.

The download button enables downloading the infered tree in newick format (tree.newick).
'''

st.markdown('#### Limitations')
'''
The tool is designed to process rooted, binary trees, oriented verically or horizontally, with all lines nearly horizontal or vertical.
If a tree is horizontal, the leaf labels need to be aligned vertically and located on the right of the tree (root needs to be on the left).
If a tree is vertical, the leaf labels need to be aligned horizontally and located on the bottom of the tree (root needs to be on the top).
The tool is not applicable to trees with multifurcations or textual data other than leaf labels.
The method does not account for branch lengths.
Keep in mind that some images require parameter tuning for successful labels and nodes detection.
Nodes detection can be time-consuming for higher image resolution.
For images containing short lines, corners may be detected as internal nodes, if ```instersection threshold``` is too low.
'''