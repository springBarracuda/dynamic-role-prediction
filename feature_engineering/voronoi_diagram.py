import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon


class VoronoiDiagram:
    def __init__(self, bounding_box=(0, 120, 0, 53.3)):
        self.bounding_box = bounding_box
        self.vor = None

    def compute_voronoi_areas(self, points):
        # Filter points inside bounding box
        x_min, x_max, y_min, y_max = self.bounding_box
        in_box = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        filtered_points = points[in_box]

        # Add mirrored points for bounded Voronoi
        mirrored = np.vstack([
            filtered_points, 
            np.c_[2 * x_min - filtered_points[:, 0], filtered_points[:, 1]],
            np.c_[2 * x_max - filtered_points[:, 0], filtered_points[:, 1]],
            np.c_[filtered_points[:, 0], 2 * y_min - filtered_points[:, 1]],
            np.c_[filtered_points[:, 0], 2 * y_max - filtered_points[:, 1]]
        ])

        # Compute Voronoi
        self.vor = Voronoi(mirrored)
        box_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

        # Calculate areas
        areas = np.zeros(len(points))
        for i, region_index in enumerate(self.vor.point_region[:len(filtered_points)]):
            region = self.vor.regions[region_index]
            if -1 in region or not region:
                continue
            poly = Polygon(self.vor.vertices[region])
            clipped = poly.intersection(box_polygon)
            areas[np.where(in_box)[0][i]] = clipped.area if not clipped.is_empty else 0

        return areas

    def plot_voronoi(self):
        """
        Plots the Voronoi diagram for the input points within the bounding box.
        """
        if self.vor is None:
            raise ValueError("Voronoi diagram has not been computed. Run 'compute_voronoi_areas' first.")
        
        fig, ax = plt.subplots()
        voronoi_plot_2d(self.vor, ax=ax)

        # Highlight bounding box and restrict plot limits
        ax.plot([self.bounding_box[0], self.bounding_box[1], self.bounding_box[1], self.bounding_box[0], self.bounding_box[0]],
                [self.bounding_box[2], self.bounding_box[2], self.bounding_box[3], self.bounding_box[3], self.bounding_box[2]],
                'k-', lw=2)
        plt.xlim(self.bounding_box[0], self.bounding_box[1])
        plt.ylim(self.bounding_box[2], self.bounding_box[3])
        plt.title("Voronoi Diagram within Bounding Box")
        plt.show()

# Example points and bounding box
# points = np.random.rand(10, 2) * [120, 53.3]
# bounding_box = (0, 120, 0, 53.3)

# Initialize the Voronoi diagram object
# voronoi_diagram = VoronoiDiagram(bounding_box)

# Compute Voronoi areas within the bounding box
# areas = voronoi_diagram.compute_voronoi_areas(points)
# print("Voronoi region areas within the bounding box:", areas)
# print("Sum of Voronoi Areas: ", sum(areas))
# print("Actual Sum: ", 120*53.3)

# Plot the Voronoi diagram within the bounding box
# voronoi_diagram.plot_voronoi()
