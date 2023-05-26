import cv2
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from shapely.geometry import LineString


class Image:
    def __init__(
            self,
            image_path,
            orientation="horizontal",
    ):
        self.image_path = image_path
        self.orientation = orientation

        self.BnW_image = self.convert_to_bnw()
        self.nonzero_pixels = self.get_nonzero_pixels()

    def convert_to_bnw(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, BnW_image) = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
        BnW_image = np.where(BnW_image == 255, 0, 255)
        return BnW_image

    def get_nonzero_pixels(self):
        nonzero_pixels = []
        for y, x in zip(list(self.BnW_image.nonzero()[0]), list(self.BnW_image.nonzero()[1])):
            nonzero_pixels.append((x, y))
        return nonzero_pixels

    def get_top_coordinates(
            self,
            coord="y",
            min_freq=10,
    ):
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
            max_gap=2,
            min_line_length=20,
            min_freq=50,
    ):
        y_candidates = self.get_top_coordinates(coord='y', min_freq=min_freq)
        x_candidates = self.get_top_coordinates(coord='x', min_freq=min_freq)

        vertical_lines, horizontal_lines = [], [] # {}, {}
        for y_candidate in y_candidates:
            # horizontal_lines[y_candidate] = self.find_line_coords_y(y_candidate, max_gap, min_line_length)
            horizontal_lines.extend(self.find_line_coords_y(y_candidate, max_gap, min_line_length))

        for x_candidate in x_candidates:
            # vertical_lines[x_candidate] = self.find_line_coords_x(x_candidate, max_gap, min_line_length)
            vertical_lines.extend(self.find_line_coords_x(x_candidate, max_gap, min_line_length))

        return vertical_lines, horizontal_lines

    def plot_lines(
            self,
            lines,
            figsize=(10, 6)
    ):
        fig = plt.figure(figsize=figsize)
        for line in lines:
            if line:
                x1, y1, x2, y2 = line
                # x2, y2 = line[0][1]
                plt.plot([x1, x2], [y1, y2], marker='o', color='b')
        print(f"Here = {self.BnW_image.shape}")
        xmin, xmax, ymin, ymax = 0, self.BnW_image.shape[0], 0, self.BnW_image.shape[1]
        plt.ylim(ymax, ymin)
        plt.show()
        return fig

    def find_lines_intersections_leaves(self, filter=10, prolongue=2):
        v_lines, h_lines = self.find_all_lines(max_gap=2, min_line_length=20, min_freq=10)
        all_lines = v_lines + h_lines
        fig = self.plot_lines(all_lines)
        intersections = find_all_intersections(v_lines, h_lines, prolongue=prolongue)
        plot_lines_and_intersections(all_lines, intersections, title='Lines and all intersections')
        filtered_intersections = filter_intersections(intersections, filter=filter)
        plot_lines_and_intersections(all_lines, filtered_intersections, title='Lines and filtered intersections')
        leaves_lines, leaves_points = find_leaves(v_lines, t=5)
        filtered_leaves = filter_intersections(leaves_points, filter=filter)
        plot_lines_and_intersections(all_lines, filtered_intersections+filtered_leaves, title='Lines and filtered intersections and leaves')
        return v_lines, h_lines, filtered_intersections, filtered_leaves

def find_intersection(v_line, h_line):
    linestring1 = LineString([tuple(v_line[:2]), tuple(v_line[2:])])
    linestring2 = LineString([tuple(h_line[:2]), tuple(h_line[2:])])
    int_pt = linestring1.intersection(linestring2)
    if int_pt.is_empty:
        return None
    else:
        return (int_pt.x, int_pt.y)
    
def prolongue_v_line(v_line, prolongue):
    x1, y1, x2, y2 = v_line
    if y1<y2:
        return (x1, y1-prolongue, x2, y2+prolongue)
    else:
        return (x1, y1+prolongue, x2, y2-prolongue)
    
def prolongue_h_line(h_line, prolongue):
    x1, y1, x2, y2 = h_line
    if x1<x2:
        return (x1-prolongue, y1, x2+prolongue, y2)
    else:
        return (x1+prolongue, y1, x2-prolongue, y2)
    

def find_all_intersections(v_lines, h_lines, prolongue):
    intersections = []
    for v_line in v_lines:
        v_line_longer = prolongue_v_line(v_line, prolongue)
        for h_line in h_lines:
            h_line_longer = prolongue_h_line(h_line, prolongue)
            intersection = find_intersection(v_line_longer, h_line_longer)
            if intersection is not None:
                intersections.append(intersection)
    return intersections
        
def filter_intersections(intersections, filter):
    """
    t - stands for treshold
    """
    filtered_intersections = []
    seen = defaultdict(lambda: False)
    print(f"Initial number of interesections = {len(intersections)}")
    
    for inter in intersections:
        if not seen[inter]:
            x, y = inter
            similar = [i for i in intersections if x-filter <= i[0] <= x+filter and y-filter <= i[1] <= y+filter]
            if similar:
                filtered_intersections.append(tuple(np.mean(similar, axis=0)))
            for similar_intersection in similar:
                seen[similar_intersection] = True
                
    print(f"After filtering number of interesections = {len(filtered_intersections)}")
    return filtered_intersections

def plot_lines_and_intersections(
    lines, 
    intersections, 
    figsize=(10, 6), 
    legend=None,
    title="Lines and intersections"
):
    plt.figure(figsize=figsize)    
    for line in lines:
        x = [line[0], line[2]]
        y = [line[1], line[3]]
        plt.plot(x, y, 'b-')
    for i, point in enumerate(intersections):
        if legend:
            plt.plot(point[0], point[1], "o", label="Point " + str(i), markersize=10)
        else:
            plt.plot(point[0], point[1], "ro")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymax, ymin)
    if legend:
        plt.legend()
    plt.title(title)
    plt.show()

def find_leaves(v_lines, t=5):
    v_lines_sorted = sorted(v_lines, key = lambda x: max(x[1], x[3]), reverse=True)
    max_y = max(v_lines_sorted[0][1], v_lines_sorted[0][3])
    
    leaves_points, leaves_lines = [], []
    for line in v_lines_sorted:
        x1, y1, x2, y2 = line
        if abs(max(y1, y2)- max_y) <= t:
            leaves_lines.append(line)
            max_ind = np.argmax([y1, y2])*2
            point = (line[max_ind], max(y1, y2))
            leaves_points.append(point)
        else:
            break
    return leaves_lines, leaves_points