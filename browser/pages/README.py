import streamlit as st

st.title("Manual")

st.markdown('#### Pipeline description')

st.markdown('#### Parameters description and tuning')
"""
* ```orientation``` - 'horizontal' or 'vertical' - orientation of the input tree:
    * horizontal if leaf labels are aligned vertically on the right hand side of the image,
    * vertical if leaf labels are aligned horizontally at the bottom of the image.
* ```minimum frequency``` - int, default to 100 - as mentioned in pipeline description, we detect horizontal and vertical lines based on the number of non-zero pixels aligned horizontally/vertically in an array-representation of the image; for this purpose we consider only the rows/columns where the number of non-zero pixels is greater or equal ```minimum frequency```; if there are short lines in the input image that are not detected, try decreasing ```minimum frequency```, but this may increase the computation time.
* ```instersection threshold``` - int - while detecting internal nodes, we consider all intersections between horizontal and vertical lines; we need to filter out the corners (intersections close to end of both of lines);  if for one of the intersecting lines, the distances (in pixels) from both ends to the intersection are greater than  ```instersection threshold```, then we consider this intersection as an internal node candidate;  if some corners are detected as internal nodes, try increasing ```instersection threshold```.
* ```resize factor``` - float, default to 0.25 - scaling factor of the input image after text detection and removal, used for coping with varying resolution of input images;  increasing ```resize factor``` may thicken the detected lines, consider doing it if some lines are thin.

"""

st.markdown('#### Output description')

st.markdown('#### Limitations')
