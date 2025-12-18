"""
Łączenie ścieżek - algorytm Dijkstry, wyszukiwanie połączeń między odcinkami.
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional

from clusters import label_with_diagonals
from path_analysis import calculate_pixel_cost, get_direction_vector, check_pixel_threshold


def find_endpoints(mask: np.ndarray, min_neighbors: int = 1, max_neighbors: int = 2) -> list:
    """
    Znajduje punkty końcowe ścieżek w masce.

    Args:
        mask: Maska binarna
        min_neighbors: Minimalna liczba sąsiadów
        max_neighbors: Maksymalna liczba sąsiadów

    Returns:
        Lista punktów końcowych (row, col)
    """
    height, width = mask.shape
    endpoints = []

    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0), (1, 1)]

    white_pixels = np.argwhere(mask)

    for row, col in white_pixels:
        neighbor_count = 0
        for dr, dc in neighbors_offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                neighbor_count += 1

        if min_neighbors <= neighbor_count <= max_neighbors:
            endpoints.append((row, col))

    return endpoints


def dijkstra_path_between_endpoints(start: tuple, end: tuple, mask: np.ndarray,
                                    bands: np.ndarray, arr: np.ndarray, ndvi: np.ndarray,
                                    ndvi_min: float, ndvi_max: float, max_distance: int,
                                    loose_mode: bool = False) -> tuple:
    """
    Znajduje najkrótszą ścieżkę między dwoma punktami końcowymi używając algorytmu Dijkstry.

    Args:
        start: Punkt startowy (row, col)
        end: Punkt końcowy (row, col)
        mask: Maska binarna
        bands: Pasma spektralne
        arr: Statystyki spektralne
        ndvi: NDVI
        ndvi_min/ndvi_max: Progi NDVI
        max_distance: Maksymalna odległość
        loose_mode: Tryb luźniejszej oceny

    Returns:
        Tuple (ścieżka, średni_koszt) lub (None, inf)
    """
    height, width = mask.shape
    start_row, start_col = start
    end_row, end_col = end

    # Sprawdź odległość euklidesową
    euclidean_dist = np.sqrt((end_row - start_row)**2 + (end_col - start_col)**2)
    if euclidean_dist > max_distance:
        return None, float('inf')

    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0), (1, 1)]

    # Dijkstra
    heap = [(0.0, start_row, start_col, [(start_row, start_col)])]
    visited = set()
    costs = {(start_row, start_col): 0.0}

    while heap:
        current_cost, row, col, path = heapq.heappop(heap)

        if (row, col) in visited:
            continue
        visited.add((row, col))

        # Sprawdź czy dotarliśmy do celu
        if row == end_row and col == end_col:
            avg_cost = current_cost / max(len(path), 1)
            return path, avg_cost

        # Ogranicz obszar przeszukiwania
        if len(path) > max_distance * 2:
            continue

        for dr, dc in neighbors_offsets:
            nr, nc = row + dr, col + dc

            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if (nr, nc) in visited:
                continue

            # Koszt ruchu po przekątnej jest większy
            move_cost = 1.414 if dr != 0 and dc != 0 else 1.0

            # Oblicz koszt piksela docelowego
            if mask[nr, nc]:
                pixel_cost = 0.1
            else:
                pixel_cost = calculate_pixel_cost(nr, nc, bands, arr, ndvi, ndvi_min, ndvi_max, loose_mode=loose_mode)
                dist_to_end = np.sqrt((end_row - nr)**2 + (end_col - nc)**2)
                penalty = 0.005 if loose_mode else 0.01
                pixel_cost += dist_to_end * penalty

            new_cost = current_cost + move_cost * (1.0 + pixel_cost)

            if (nr, nc) not in costs or new_cost < costs[(nr, nc)]:
                costs[(nr, nc)] = new_cost
                new_path = path + [(nr, nc)]
                heapq.heappush(heap, (new_cost, nr, nc, new_path))

    return None, float('inf')


def connect_paths_dijkstra(mask: np.ndarray, bands: np.ndarray, arr: np.ndarray,
                           ndvi: np.ndarray, ndvi_min: float, ndvi_max: float,
                           max_distance: int = 50, cost_threshold: float = 5.0,
                           min_cluster_depth: int = 50, loose_mode: bool = False) -> tuple:
    """
    Łączy ścieżki poprzez wyszukiwanie połączeń między punktami końcowymi.

    Args:
        mask: Maska binarna
        bands: Pasma spektralne
        arr: Statystyki spektralne
        ndvi: NDVI
        ndvi_min/ndvi_max: Progi NDVI
        max_distance: Maksymalna odległość połączenia
        cost_threshold: Próg kosztu
        min_cluster_depth: Minimalna głębokość klastra
        loose_mode: Tryb luźniejszej oceny

    Returns:
        Tuple (rozszerzona_maska, liczba_połączeń)
    """
    print("  Znajdowanie punktów końcowych...")
    endpoints = find_endpoints(mask, min_neighbors=1, max_neighbors=2)
    print(f"  Znaleziono {len(endpoints)} punktów końcowych")

    if len(endpoints) < 2:
        return mask, 0

    # Znajdź klastry i ich głębokości
    labeled_img, num_features = label_with_diagonals(mask)
    cluster_depths = {}

    # Przypisz każdy endpoint do klastra
    endpoint_clusters = {}
    for ep in endpoints:
        cluster_id = labeled_img[ep[0], ep[1]]
        endpoint_clusters[ep] = cluster_id
        if cluster_id not in cluster_depths:
            cluster_depths[cluster_id] = np.sum(labeled_img == cluster_id)

    # Filtruj endpointy z małych klastrów
    min_depth = min_cluster_depth // 2 if loose_mode else min_cluster_depth
    filtered_endpoints = [ep for ep in endpoints
                          if cluster_depths.get(endpoint_clusters[ep], 0) >= min_depth]
    print(f"  Po filtracji: {len(filtered_endpoints)} punktów końcowych")

    if len(filtered_endpoints) < 2:
        return mask, 0

    expanded_mask = mask.copy()
    connections_made = 0

    filtered_endpoints.sort(key=lambda x: (x[0], x[1]))

    effective_threshold = cost_threshold * 1.5 if loose_mode else cost_threshold

    for i, ep1 in enumerate(filtered_endpoints):
        cluster1 = endpoint_clusters[ep1]

        for ep2 in filtered_endpoints[i+1:]:
            cluster2 = endpoint_clusters[ep2]

            if cluster1 == cluster2:
                continue

            dist = np.sqrt((ep2[0] - ep1[0])**2 + (ep2[1] - ep1[1])**2)
            if dist > max_distance:
                continue

            path, avg_cost = dijkstra_path_between_endpoints(
                ep1, ep2, expanded_mask, bands, arr, ndvi, ndvi_min, ndvi_max, max_distance, loose_mode=loose_mode
            )

            if path is not None and avg_cost < effective_threshold:
                for row, col in path:
                    expanded_mask[row, col] = True
                connections_made += 1
                print(f"    Połączono: {ep1} -> {ep2} (koszt: {avg_cost:.2f}, długość: {len(path)})")

    added_pixels = np.sum(expanded_mask) - np.sum(mask)
    print(f"  Utworzono {connections_made} połączeń, dodano {added_pixels} pikseli")

    return expanded_mask, connections_made


def find_vectors_in_cone(endpoint: tuple, direction: tuple, all_clusters: list,
                         current_cluster, cone_angle: int = 45,
                         max_distance: int = 150) -> list:
    """
    Szuka końców innych wektorów w stożku przed punktem końcowym.

    Args:
        endpoint: Punkt końcowy
        direction: Kierunek
        all_clusters: Lista wszystkich klastrów
        current_cluster: Aktualny klaster
        cone_angle: Kąt stożka w stopniach
        max_distance: Maksymalna odległość

    Returns:
        Lista (klaster, punkt_końcowy, odległość, kąt, indeks_końca)
    """
    candidates = []
    cone_cos = math.cos(math.radians(cone_angle))

    for cluster in all_clusters:
        if cluster is current_cluster:
            continue

        path = cluster.longest_path
        if path is None or len(path) < 10:
            continue

        # Sprawdź oba końce ścieżki
        for end_idx in [0, -1]:
            other_end = path[end_idx]

            dist = math.sqrt((other_end[0] - endpoint[0])**2 +
                             (other_end[1] - endpoint[1])**2)

            if dist > max_distance or dist < 5:
                continue

            dir_to_other = (
                (other_end[0] - endpoint[0]) / dist,
                (other_end[1] - endpoint[1]) / dist
            )

            dot = direction[0] * dir_to_other[0] + direction[1] * dir_to_other[1]

            if dot >= cone_cos:
                angle = math.degrees(math.acos(max(-1, min(1, dot))))
                candidates.append((cluster, other_end, dist, angle, end_idx))

    candidates.sort(key=lambda x: x[2])
    return candidates


def connect_clusters_with_pixels(start_point: tuple, end_point: tuple, mask: np.ndarray,
                                 bands: np.ndarray, arr: np.ndarray, ndvi: np.ndarray,
                                 ndvi_min: float, ndvi_max: float,
                                 min_bands_match: int = 5) -> Optional[list]:
    """
    Próbuje połączyć dwa punkty przez "zdatne" piksele.

    Args:
        start_point: Punkt startowy
        end_point: Punkt końcowy
        mask: Maska binarna
        bands: Pasma spektralne
        arr: Statystyki spektralne
        ndvi: NDVI
        ndvi_min/ndvi_max: Progi NDVI
        min_bands_match: Minimalna liczba pasujących pasm

    Returns:
        Lista pikseli łączących lub None
    """
    height, width = mask.shape
    path = []
    current = start_point
    target = end_point
    visited = {start_point}

    max_steps = int(math.sqrt((end_point[0]-start_point[0])**2 +
                              (end_point[1]-start_point[1])**2) * 2)

    for _ in range(max_steps):
        if current == target:
            return path

        best_neighbor = None
        best_score = float('inf')

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = current[0] + dr, current[1] + dc

                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if (nr, nc) in visited:
                    continue

                dist_to_target = math.sqrt((target[0]-nr)**2 + (target[1]-nc)**2)

                is_valid = check_pixel_threshold(nr, nc, bands, arr, 0.80, 1.20,
                                                 ndvi, ndvi_min, ndvi_max, min_bands_match)

                if mask[nr, nc]:
                    score = dist_to_target - 100
                elif is_valid:
                    score = dist_to_target
                else:
                    score = dist_to_target + 50

                if score < best_score:
                    best_score = score
                    best_neighbor = (nr, nc)

        if best_neighbor is None:
            return None

        visited.add(best_neighbor)
        if not mask[best_neighbor[0], best_neighbor[1]]:
            path.append(best_neighbor)
        current = best_neighbor

        if math.sqrt((target[0]-current[0])**2 + (target[1]-current[1])**2) < 3:
            return path

    return None


def check_connection_angle(cluster1, end_idx1: int, cluster2, end_idx2: int,
                           max_angle: int = 80) -> tuple:
    """
    Sprawdza czy połączenie dwóch klastrów nie tworzy zbyt ostrego kąta.

    Args:
        cluster1: Pierwszy klaster
        end_idx1: Indeks końca pierwszego klastra
        cluster2: Drugi klaster
        end_idx2: Indeks końca drugiego klastra
        max_angle: Maksymalny kąt

    Returns:
        Tuple (czy_akceptowalny, kąt_połączenia)
    """
    path1 = cluster1.longest_path
    path2 = cluster2.longest_path

    sample_len = min(20, len(path1)//4)
    if end_idx1 == 0:
        dir1 = get_direction_vector(path1, from_end=False, sample_length=sample_len)
    else:
        dir1 = get_direction_vector(path1, from_end=True, sample_length=sample_len)

    sample_len = min(20, len(path2)//4)
    if end_idx2 == 0:
        dir2_raw = get_direction_vector(path2, from_end=False, sample_length=sample_len)
        dir2 = (-dir2_raw[0], -dir2_raw[1])
    else:
        dir2_raw = get_direction_vector(path2, from_end=True, sample_length=sample_len)
        dir2 = (-dir2_raw[0], -dir2_raw[1])

    dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    dot = max(-1, min(1, dot))
    angle = math.degrees(math.acos(dot))

    return angle >= (180 - max_angle), angle

