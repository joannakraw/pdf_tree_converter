import cv2
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import defaultdict
from shapely.geometry import LineString, Point


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
        lines_coords = [[(line[0], candidate), (line[-1], candidate)] for line in lines]
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
        lines_coords = [[(candidate, line[0]), (candidate, line[-1])] for line in lines]
        return lines_coords

    def find_all_lines(
            self,
            max_gap=2,
            min_line_length=20,
            min_freq=50,
    ):
        y_candidates = self.get_top_coordinates(coord='y', min_freq=min_freq)
        x_candidates = self.get_top_coordinates(coord='x', min_freq=min_freq)

        vertical_lines, horizontal_lines = {}, {}
        for y_candidate in y_candidates:
            horizontal_lines[y_candidate] = self.find_line_coords_y(y_candidate, max_gap, min_line_length)

        for x_candidate in x_candidates:
            vertical_lines[x_candidate] = self.find_line_coords_x(x_candidate, max_gap, min_line_length)

        return vertical_lines, horizontal_lines

    def merge_lines(
            self,
    ):
        pass

    def plot_lines(
            self,
            lines,
            figsize=(10, 6)
    ):
        fig = plt.figure(figsize=figsize)
        for line in lines:
            if line:
                x1, y1 = line[0][0]
                x2, y2 = line[0][1]
                plt.plot([x1, x2], [y1, y2], marker='o', color='b')
        print(f"Here = {self.BnW_image.shape}")
        xmin, xmax, ymin, ymax = 0, self.BnW_image.shape[0], 0, self.BnW_image.shape[1]
        plt.ylim(ymax, ymin)
        plt.show()
        return fig

