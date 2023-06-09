import cv2
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
from shapely.geometry import LineString
import matplotlib as mpl
from easyocr import Reader

reader = Reader(lang_list=["en"])

def convert_to_bnw(image):
    """
    Reads an image from a given path and converts it to black and white colorscale.
    @param path: path to image file
    @return: an image converted to black and white scale
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, BnW_image) = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
    BnW_image = np.where(BnW_image > 200, 0, 255)
    return BnW_image


class TreeImage:
    def __init__(
            self,
            image_path,
            resize_factor=1,
            orientation="horizontal",
            preprocessed=False
    ):
        """
        Class for coping with tree images - for labels detection and text removal
        :param image_path: path to image file
        :param resize_factor: scaling factor of the input image
        :param orientation: {'horizontal' or 'vertical'}, orientation of a tree
        :param preprocessed: if the image is already preprocessed (is in black and white scale)
        """
        self.image_path = image_path
        self.orientation = orientation
        self.BnW_image = cv2.imread(self.image_path)
        if resize_factor != 1:
            self.BnW_image = cv2.resize(self.BnW_image, None, fx=resize_factor, fy=resize_factor)
        if preprocessed:            
            self.BnW_image = np.array(self.BnW_image)
            self.BnW_image = np.where(self.BnW_image > 200, 0, 255)
        else:
            self.BnW_image = convert_to_bnw(self.BnW_image)
        self.nonzero_pixels = self.get_nonzero_pixels()

    def get_nonzero_pixels(self):
        """
        Iterates over nonzero pixels from image numpy array
        @return: a list of tuples of nonzero pixels
        """
        nonzero_pixels = []
        for y, x in zip(list(self.BnW_image.nonzero()[0]), list(self.BnW_image.nonzero()[1])):
            nonzero_pixels.append((x, y))
        return nonzero_pixels

    def get_top_coordinates(
            self,
            coord,
            min_freq
    ):
        """
        Finds most frequently occurring values x and y from nonzero pixels
        @param coord: should be equal to "x" or "y", for which coordinate
        @param min_freq: minimum number of occurrences of a point
        @return: list of x or y points occurring > min_freq times in an image
        """
        if coord == "x":
            occurrences = Counter([x for x, y in self.nonzero_pixels])
        elif coord == "y":
            occurrences = Counter([y for x, y in self.nonzero_pixels])
        selected_points = [point for point, freq in occurrences.items() if freq >= min_freq]
        print(f"Number of {coord} points with frequency > {min_freq}: {len(selected_points)}")
        return selected_points

    def find_line_coords_y(
            self,
            candidate,
            max_gap=2,
            min_line_length=20
    ):
        """
        Find line coordinates (start and end) for a horizontal line with y equal to candidate
        :param candidate: value of y in the line
        :param max_gap: maximum gap in the line to be still considered as one line
        :param min_line_length: minimum line length
        :return: list of horizontal line coordinates (start, end) with y = candidate
        """
        x_candidates = sorted([x for x, y in self.nonzero_pixels if y == candidate])
        lines = []
        line = [x_candidates[0]]
        for x0, x1 in zip(x_candidates[:-1], x_candidates[1:]):
            if x1 - x0 <= max_gap:
                line.append(x1)
            else:
                line = []
            if len(line) >= min_line_length and line not in lines:
                lines.append(line)
        lines_coords = [(line[0], candidate, line[-1], candidate) for line in lines]
        return lines_coords

    def find_line_coords_x(
            self,
            candidate,
            max_gap=2,
            min_line_length=20
    ):
        """
        Find line coordinates (start and end) for a vertical line with x equal to candidate
        :param candidate: value of x in the line
        :param max_gap: maximum gap in the line to be still considered as one line
        :param min_line_length: minimum line length
        :return: list of vertical line coordinates (start, end) with x = candidate
        """
        y_candidates = sorted([y for x, y in self.nonzero_pixels if x == candidate])
        lines = []
        line = [y_candidates[0]]
        for y0, y1 in zip(y_candidates[:-1], y_candidates[1:]):
            if y1 - y0 <= max_gap:
                line.append(y1)
            else:
                line = []
            if len(line) >= min_line_length and line not in lines:
                lines.append(line)
        lines_coords = [(candidate, line[0], candidate, line[-1]) for line in lines]
        return lines_coords

    def find_all_lines(
            self,
            max_gap,
            min_line_length,
            min_freq,
    ):
        y_candidates = self.get_top_coordinates(coord='y', min_freq=min_freq)
        x_candidates = self.get_top_coordinates(coord='x', min_freq=min_freq)

        vertical_lines, horizontal_lines = [], []
        for y_candidate in y_candidates:
            horizontal_lines.extend(self.find_line_coords_y(y_candidate, max_gap, min_line_length))

        for x_candidate in x_candidates:
            vertical_lines.extend(self.find_line_coords_x(x_candidate, max_gap, min_line_length))

        return vertical_lines, horizontal_lines

    def plot_lines(
            self,
            lines,
            figsize=(10, 6),
    ):
        """
        Plots a list of lines.
        @param lines: list of lines, where each line is represented as x1, y1, x2, y2
        @param figsize: figure size
        @return: matplotlib plot with selected lines
        """
        fig = plt.figure(figsize=figsize)
        for line in lines:
            if line:
                x1, y1, x2, y2 = line
                plt.plot([x1, x2], [y1, y2], marker='o', color='b')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.ylim(ymax, ymin)
        return fig

    def find_lines_intersections_leaves(
            self,
            filter=10,
            prolong=2,
            max_gap=2,
            min_line_length=20,
            min_freq=10,
            leaves_threshold=5,
            intersection_threshold=5,
            legend=None,
            orientation='horizontal',
    ):
        # Find all lines
        v_lines, h_lines = self.find_all_lines(max_gap=max_gap,
                                               min_line_length=min_line_length,
                                               min_freq=min_freq)
        # Add horizontal and vertical lines and plot them
        all_lines = v_lines + h_lines
        self.plot_lines(all_lines)

        intersections = find_all_intersections(v_lines, h_lines, prolong=prolong, t=intersection_threshold)
        print("[Step 1] finding and plotting all intersections")
        plot_lines_and_intersections(all_lines, intersections,
                                     title='Lines and all intersections')

        print("[Step 2] filtering internal nodes (line intersections)")
        filtered_intersections = filter_intersections(intersections, filter=filter)
        plot_lines_and_intersections(all_lines, filtered_intersections,
                                     title='Lines and filtered intersections')

        if orientation == "horizontal":
            leaves_lines, leaves_points = find_leaves_horizontal(h_lines, t=leaves_threshold)

        if orientation == "vertical":
            leaves_lines, leaves_points = find_leaves_vertical(v_lines, t=leaves_threshold)

        print("[Step 3] filtering leaves (line endings)")
        filtered_leaves = filter_intersections(leaves_points, filter=filter)
        fig = plot_lines_nodes_leaves(all_lines, nodes=filtered_intersections, leaves=filtered_leaves, legend=legend)
        print(f"[Summary] number of leaves = {len(filtered_leaves)}, "
              f"number of nodes = {len(filtered_intersections)}")

        return v_lines, h_lines, filtered_intersections, filtered_leaves, fig


class Image:
    tree_image: TreeImage
    labels: list
    boxes: list

    def __init__(
            self,
            image_path,
            orientation="horizontal",
            resize_factor=1
    ):
        tree_image_path = image_path[:-4]+'_tree'+image_path[-4:]
        print(f'Tree image will be saved to {tree_image_path}.')
        self.labels, self.boxes, self.fig_boxes = self.split_tree_and_labels(image_path, tree_image_path)
        self.tree_image = TreeImage(image_path=tree_image_path, orientation=orientation,
                                    preprocessed=True, resize_factor=resize_factor)

    def split_tree_and_labels(self, image_path, tree_image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=None, fx=2, fy=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU |cv2.THRESH_BINARY_INV)
        if thresh1[0,0]<200:
            thresh1 = np.where(thresh1>200, 0, 255)
        
        results = reader.readtext(img, min_size=5, 
                                  mag_ratio=2, slope_ths=0.2)
        labels = [i[1] for i in results if i[1]]
        print('Retrieved labels:', *labels, sep=' ')

        boxes = preprocess_boxes(results)
        fig_boxes = plot_image_with_boxes(thresh1, boxes, figsize=(16, 9))
        for box in boxes:
            (x, y), w, h = box
            img[y:(y+h), x:(x+w)] = 255
        cv2.imwrite(tree_image_path, img)
        
        return labels, boxes, fig_boxes

def plot_image_with_boxes(image, boxes, figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=plt.cm.gray)
    for box in boxes:
        starting_point, width, height = box
        rect = patches.Rectangle(starting_point, width, height, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    return fig

def preprocess_boxes(results):
    boxes = []
    for result in results:
        box = result[0]
        starting_point = tuple(box[0])
        width = box[1][0] - box[0][0]
        height = box[2][1] - box[0][1]
        box = [starting_point, width, height]
        boxes.append(box)
    return boxes


def find_intersection(v_line, h_line, t=5):
    """
    Find intersection between horizontal and vertical line.
    An intersection is a common point of two lines.
    A point lying closer than t to an endpoint of each of the two intersecting lines
    is not considered an intersection.
    @param v_line: first line (vertical)
    @param h_line: second line (horizontal)
    @param t: intersection threshold
    @return: tuple with intersection point if any else None
    """
    linestring1 = LineString([tuple(v_line[:2]), tuple(v_line[2:])])
    linestring2 = LineString([tuple(h_line[:2]), tuple(h_line[2:])])
    int_pt = linestring1.intersection(linestring2)
    if int_pt.is_empty:
        return None
    if (abs(v_line[1] - int_pt.y) > t and abs(v_line[3] - int_pt.y) > t) or (abs(h_line[0] - int_pt.x) > t and abs(h_line[2] - int_pt.x) > t):
        return int_pt.x, int_pt.y
    else:
        return None


def prolong_v_line(v_line, prolong):
    """
    Prolong a vertical line by prolong number of pixels
    :param v_line: vertical line coordinates
    :param prolong: number of pixels to prolong a line
    :return: coordinates of an elongated line
    """
    x1, y1, x2, y2 = v_line
    if y1 < y2:
        return x1, y1 - prolong, x2, y2 + prolong
    else:
        return x1, y1 + prolong, x2, y2 - prolong


def prolong_h_line(h_line, prolong):
    """
    Prolong a horizontal line by prolong number of pixels
    :param h_line: horizontal line coordinates
    :param prolong: number of pixels to prolong a line
    :return: coordinates of an elongated line
    """

    x1, y1, x2, y2 = h_line
    if x1 < x2:
        return x1 - prolong, y1, x2 + prolong, y2
    else:
        return x1 + prolong, y1, x2 - prolong, y2
    

def find_all_intersections(v_lines, h_lines, prolong, t=5):
    intersections = []
    for v_line in v_lines:
        v_line_longer = prolong_v_line(v_line, prolong)
        for h_line in h_lines:
            h_line_longer = prolong_h_line(h_line, prolong)
            intersection = find_intersection(v_line_longer, h_line_longer, t=t)
            if intersection is not None:
                # print("A", v_line_longer, h_line, intersection)
                intersections.append(intersection)
    return intersections


def filter_intersections(intersections, filter):
    """
    Filter intersections - find groups of intersections lying in proximity of filter
    pixels from each other and replace them with one intersection with mean coordinates.
    :param intersections: list of intersections
    :param filter: int, parameter used to consider two intersections as neighbouring
    :return:list of filtered intersections (replaced by group's means)
    """
    filtered_intersections = []
    seen = defaultdict(lambda: False)
    print(f"Initial number of intersections = {len(intersections)}")
    
    for inter in intersections:
        if not seen[inter]:
            x, y = inter
            similar = [i for i in intersections if x-filter <= i[0] <= x+filter and y-filter <= i[1] <= y+filter]
            if similar:
                filtered_intersections.append(tuple(np.mean(similar, axis=0)))
            for similar_intersection in similar:
                seen[similar_intersection] = True
                
    print(f"After filtering number of intersections = {len(filtered_intersections)}")
    return filtered_intersections


def plot_lines_and_intersections(
    lines, 
    intersections, 
    figsize=(10, 6), 
    legend=None,
    title="Lines and intersections"
):
    """
    Plots tree branches (lines) with lines intersections
    :param lines: list of line coordinates
    :param intersections: list of intersecting points
    :param figsize: figure size
    :param legend: if you want to add legend
    :param title: plot title
    :return: a matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    for line in lines:
        x = [line[0], line[2]]
        y = [line[1], line[3]]
        plt.plot(x, y, 'b-')
    for i, point in enumerate(intersections):
        if legend:
            plt.plot(point[0], point[1], "o", label="Point " + str(point), markersize=10)
        else:
            plt.plot(point[0], point[1], "ro")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymax, ymin)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    return fig


def plot_lines_nodes_leaves(
        lines,
        leaves,
        nodes,
        figsize=(10, 6),
        legend=None,
        title=None
):
    """
    Plots tree branches (lines) with nodes and leaves separately.
    @param lines: list of lines
    @param leaves: list of tree leaves
    @param nodes: list of tree internal nodes
    @param figsize: figure size
    @param legend: if you want to add legend
    @param title: plot title
    @return: a matplotlib plot
    """
    mpl.style.use("seaborn-whitegrid")
    fig = plt.figure(figsize=figsize)
    for line in lines:
        x = [line[0], line[2]]
        y = [line[1], line[3]]
        plt.plot(x, y, 'k-')

    for i, point in enumerate(nodes):
        if legend:
            plt.plot(point[0], point[1], "o", label="Node " + str(i), markersize=10)
        else:
            plt.plot(point[0], point[1], "co", markersize=10)

    for i, point in enumerate(leaves):
        if legend:
            plt.plot(point[0], point[1], "D", label="Leaf " + str(i), markersize=10)
        else:
            plt.plot(point[0], point[1], "mD", markersize=10)

    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymax, ymin)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return fig


def find_leaves_vertical(v_lines, t=5):
    """
    Find line ends aligned on the bottom of the image, find maximum y coordinate of vertical lines,
    and then find all vertical line endings that are in no distant from this maximum y that t - threshold,
    and save them as leaf candidates.
    :param v_lines: list of vertical lines
    :param t: threshold
    :return: tuple with list of leaves lines and list leaves points
    """
    v_lines_sorted = sorted(v_lines, key=lambda x: max(x[1], x[3]), reverse=True)
    max_y = max(v_lines_sorted[0][1], v_lines_sorted[0][3])
    
    leaves_points, leaves_lines = [], []
    for line in v_lines_sorted:
        x1, y1, x2, y2 = line
        if abs(max(y1, y2) - max_y) <= t:
            leaves_lines.append(line)
            max_ind = np.argmax([y1, y2])*2
            point = (line[max_ind], max(y1, y2))
            leaves_points.append(point)
        else:
            break
    return leaves_lines, leaves_points


def find_leaves_horizontal(h_lines, t=5):
    """
    Find line ends aligned on the right of the image, find maximum x coordinate of horizontal lines,
    and then find all horizontal line endings that are in no distant from this maximum x that t - threshold,
    and save them as leaf candidates.
    :param h_lines: list of horizontal lines
    :param t: threshold
    :return: tuple with list of leaves lines and list leaves points
    """
    h_lines_sorted = sorted(h_lines, key=lambda x: max(x[0], x[2]), reverse=True)
    max_x = max(h_lines_sorted[0][0], h_lines_sorted[0][2])

    leaves_points, leaves_lines = [], []
    for line in h_lines_sorted:
        x1, y1, x2, y2 = line
        if abs(max(x1, x2) - max_x) <= t:
            leaves_lines.append(line)
            max_ind = np.argmax([x1, x2])
            point = (max(x1, x2), line[max_ind])
            leaves_points.append(point)
        else:
            break

    return leaves_lines, leaves_points
