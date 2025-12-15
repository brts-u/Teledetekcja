from __future__ import annotations

from typing import List, Tuple
import time
import networkx as nx
import pickle
import sys
import rasterio
import numpy as np
from scipy import ndimage


def label_with_diagonals(input_array, structure=None):
    # Convert to numpy array
    arr = np.asarray(input_array)

    # Create default 8-connectivity structure if none provided
    if structure is None:
        # For 2D arrays, use 3x3 structure with all True (8-connectivity)
        if arr.ndim == 2:
            structure = np.ones((3, 3), dtype=bool)
        # For 3D arrays, use 3x3x3 structure with all True (26-connectivity)
        elif arr.ndim == 3:
            structure = np.ones((3, 3, 3), dtype=bool)
        # For other dimensions, create appropriate structure
        else:
            structure = np.ones([3] * arr.ndim, dtype=bool)

    # Use scipy's label function with our structure
    labeled_array, num_features = ndimage.label(arr, structure=structure)

    return labeled_array, num_features

class Cluster:
    def __init__(self, cluster_id: int, cells: List[Tuple[int, int]]):
        self.cluster_id = cluster_id
        self.cells = cells
        self.depth = 0
        self.longest_path = None

    def __repr__(self):
        return f"Cluster(id={self.cluster_id}, size={self.size()} depth={self.depth})"

    def size(self) -> int:
        return len(self.cells)

    def add_cell(self, cell: Tuple[int, int]) -> None:
        self.cells.append(cell)

    def get_depth(self) -> (int, List[Tuple[int, int]]):
        # Create a graph
        G = nx.Graph()

        # Add nodes and edges for all cells in the cluster
        for cell in self.cells:
            x, y = cell
            neighbors = [
                (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y), (x + 1, y),
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
            ]
            for neighbor in neighbors:
                if neighbor in self.cells:
                    G.add_edge(cell, neighbor)

        # Start from an arbitrary node
        start = list(G.nodes())[0]

        # Find farthest from start
        lengths1 = nx.single_source_shortest_path_length(G, start)
        node1 = max(lengths1, key=lengths1.get)

        # Find farthest from node1
        lengths2 = nx.single_source_shortest_path_length(G, node1)
        node2 = max(lengths2, key=lengths2.get)

        self.depth = lengths2[node2]
        path = nx.shortest_path(G, node1, node2)
        self.longest_path = path

        return self.depth

def create_clusters(labeled_image):
    Q = {}

    for x, row in enumerate(labeled_image):
        for y, value in enumerate(row):
            if value == 0:
                continue
            try:
                Q[value].add_cell((x, y))
            except KeyError:
                Q[value] = Cluster(value, [(x, y)])
    return Q

def save_pickle_to_raster(pickle_path):

    NO_DATA_VALUE = -9999

    with open(pickle_path, 'rb') as f:
        clusters = pickle.load(f)
    all_cells = []
    for cluster in clusters.values():
        all_cells.extend(cluster.cells)

    if not all_cells:
        raise ValueError("No cells found in clusters")

    all_rows, all_cols = zip(*all_cells)
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    raster = np.full((height, width), NO_DATA_VALUE, dtype=np.int32)
    for cluster in clusters.values():
        if cluster.depth is None:
            continue
        for row, col in cluster.cells:
            raster[(row - min_row), (col - min_col)] = cluster.depth

    with rasterio.open(
            'depth_clusters.tif',
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=crs,
            transform=transform,
            nodata=NO_DATA_VALUE,
            compress='lzw'
    ) as dst:
        dst.write(raster, 1)

    print(f"GeoTIFF saved to {'depth_clusters.tif'}")
    print(f"Raster dimensions: {height} x {width}")
    print(f"Value range: {raster[raster != NO_DATA_VALUE].min()} to {raster[raster != NO_DATA_VALUE].max()}")


if __name__ == "__main__":

    with rasterio.open('complete_mask.tif') as src:
        img = src.read(1).astype(bool)  # Read as boolean
        transform = src.transform
        crs = src.crs

    assert img is not None, "file could not be read, check with os.path.exists()"
    # if os.path.exists('clusters.pkl'):
    #     save_pickle_to_raster('clusters.pkl')
    #     exit(0)

    labeled_img, num_features = label_with_diagonals(img)
    with rasterio.open('labeled_image.tif', 'w', driver='GTiff', height=labeled_img.shape[0], width=labeled_img.shape[1],
                       crs=crs, transform=transform, count=1, dtype=labeled_img.dtype) as dst:
        dst.write(labeled_img, 1)

    print("Creating clusters...")
    t0 = time.time()
    clusters = create_clusters(labeled_img)
    t1 = time.time()
    print(f"Total clusters found: {len(clusters)}, {t1-t0:.2f} seconds")

    last_percent = 0
    size_sum = 0
    for cluster in clusters:
        size_sum += clusters[cluster].size()
    print('Calculating depths for clusters (this may take a while)...')
    partial_size_sum = 0
    t0 = time.time()
    for cluster in clusters.values():
        if not cluster.size() < 100:
            depth = cluster.get_depth()
        partial_size_sum += cluster.size()
        percent = (partial_size_sum + 1) * 100 // size_sum  # Calculate progress percentage
        if percent > last_percent:  # Update only when the percentage changes
            elapsed = time.time() - t0
            sys.stdout.write(f"\rProgress: {percent}% (est. time remaining {elapsed * (100 - percent) / (percent + 1):.2f} seconds)")
            sys.stdout.flush()
            last_percent = percent

    sys.stdout.write("\nDone!\n")  # Move to the next line after the loop

    # Zapis do pickla, obliczanie depth trochę trwa
    with open('clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)
