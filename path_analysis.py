"""
Analiza ścieżek - kąty, kierunki, szerokość torów.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


def calculate_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Oblicza kąt w punkcie p2 utworzony przez punkty p1-p2-p3.

    Args:
        p1: Pierwszy punkt (row, col)
        p2: Wierzchołek kąta (row, col)
        p3: Trzeci punkt (row, col)

    Returns:
        Kąt w stopniach (0-180)
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 180.0

    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def get_direction_vector(path: list, from_end: bool = True, sample_length: int = 10) -> tuple:
    """
    Oblicza wektor kierunku na końcu ścieżki.

    Args:
        path: Lista punktów ścieżki
        from_end: True dla kierunku na końcu, False dla początku
        sample_length: Liczba pikseli do próbkowania kierunku

    Returns:
        Znormalizowany wektor (dr, dc)
    """
    if len(path) < 2:
        return (0, 0)

    sample_length = min(sample_length, len(path) - 1)

    if from_end:
        p_end = path[-1]
        p_prev = path[-1 - sample_length]
        dr = p_end[0] - p_prev[0]
        dc = p_end[1] - p_prev[1]
    else:
        p_start = path[0]
        p_next = path[sample_length]
        dr = p_start[0] - p_next[0]
        dc = p_start[1] - p_next[1]

    mag = math.sqrt(dr**2 + dc**2)
    if mag == 0:
        return (0, 0)

    return (dr / mag, dc / mag)


def check_path_angles(path: list, min_angle: int = 120, sample_step: int = 5) -> bool:
    """
    Sprawdza czy ścieżka nie zawiera ostrych kątów.

    Args:
        path: Lista punktów ścieżki
        min_angle: Minimalny dopuszczalny kąt
        sample_step: Co ile pikseli sprawdzać kąt

    Returns:
        True jeśli ścieżka jest prawidłowa (bez ostrych kątów)
    """
    if len(path) < sample_step * 2 + 1:
        return True

    for i in range(sample_step, len(path) - sample_step, sample_step):
        p1 = path[i - sample_step]
        p2 = path[i]
        p3 = path[i + sample_step]

        angle = calculate_angle(p1, p2, p3)
        if angle < min_angle:
            return False

    return True


def check_path_has_sharp_turns(path: list, max_angle: int = 80, sample_step: int = 10) -> tuple:
    """
    Sprawdza czy ścieżka ma ostre zakręty (> max_angle stopni).

    Args:
        path: Lista punktów ścieżki
        max_angle: Maksymalny kąt zakrętu
        sample_step: Co ile pikseli sprawdzać

    Returns:
        Tuple (ma_ostre_zakręty, lista_kątów)
    """
    if len(path) < sample_step * 2 + 1:
        return False, []

    angles = []
    has_sharp = False

    for i in range(sample_step, len(path) - sample_step, sample_step):
        p1 = path[i - sample_step]
        p2 = path[i]
        p3 = path[i + sample_step]

        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

        # Kąt < (180 - max_angle) oznacza zakręt > max_angle stopni
        if angle < (180 - max_angle):
            has_sharp = True

    return has_sharp, angles


def find_cluster_endpoints_with_direction(cluster, min_angle: int = 120) -> list:
    """
    Znajduje punkty końcowe klastra wraz z ich kierunkami.

    Args:
        cluster: Obiekt klastra
        min_angle: Minimalny kąt do walidacji

    Returns:
        Lista (punkt, kierunek, czy_prawidłowy)
    """
    path = cluster.longest_path
    if path is None or len(path) < 10:
        return []

    # Sprawdź kąty w całej ścieżce
    is_valid = check_path_angles(path, min_angle=min_angle)

    # Kierunek na początku (wychodzący)
    dir_start = get_direction_vector(path, from_end=False, sample_length=min(20, len(path)//4))
    # Kierunek na końcu (wychodzący)
    dir_end = get_direction_vector(path, from_end=True, sample_length=min(20, len(path)//4))

    endpoints = [
        (path[0], dir_start, is_valid),
        (path[-1], dir_end, is_valid)
    ]

    return endpoints


def check_pixel_threshold(row: int, col: int, bands: np.ndarray, arr: np.ndarray,
                          scale_min: float, scale_max: float, ndvi: np.ndarray,
                          ndvi_min: float, ndvi_max: float, min_bands_match: int = 6) -> bool:
    """
    Sprawdza czy piksel spełnia próg pewności.

    Args:
        row, col: Współrzędne piksela
        bands: Tablica pasm spektralnych (8, H, W)
        arr: Statystyki spektralne
        scale_min/scale_max: Mnożniki progów
        ndvi: Tablica NDVI
        ndvi_min/ndvi_max: Progi NDVI
        min_bands_match: Minimalna liczba pasujących pasm

    Returns:
        True jeśli co najmniej min_bands_match pasm pasuje
    """
    if row < 0 or row >= bands.shape[1] or col < 0 or col >= bands.shape[2]:
        return False

    matches = 0
    for i, (min_val, mean_val, max_val) in enumerate(arr):
        min_thresh = min_val * scale_min
        max_thresh = max_val * scale_max
        pixel_val = bands[i, row, col]
        if min_thresh <= pixel_val <= max_thresh:
            matches += 1

    # Sprawdź NDVI
    if ndvi_min <= ndvi[row, col] <= ndvi_max:
        matches += 1

    return matches >= min_bands_match


def calculate_pixel_cost(row: int, col: int, bands: np.ndarray, arr: np.ndarray,
                         ndvi: np.ndarray, ndvi_min: float, ndvi_max: float,
                         loose_mode: bool = False) -> float:
    """
    Oblicza koszt piksela - im bardziej pasuje do wzorca torów, tym niższy koszt.

    Args:
        row, col: Współrzędne piksela
        bands: Tablica pasm spektralnych
        arr: Statystyki spektralne
        ndvi: Tablica NDVI
        ndvi_min/ndvi_max: Progi NDVI
        loose_mode: Tryb luźniejszej oceny

    Returns:
        Wartość od 0 (idealnie pasuje) do wysokiej wartości (nie pasuje)
    """
    if row < 0 or row >= bands.shape[1] or col < 0 or col >= bands.shape[2]:
        return float('inf')

    total_deviation = 0.0
    range_multiplier = 1.5 if loose_mode else 1.0

    for i, (min_val, mean_val, max_val) in enumerate(arr):
        pixel_val = bands[i, row, col]
        range_val = max(max_val - min_val, 1e-6) * range_multiplier

        if pixel_val < min_val:
            deviation = (min_val - pixel_val) / range_val
        elif pixel_val > max_val:
            deviation = (pixel_val - max_val) / range_val
        else:
            deviation = 0.0

        total_deviation += deviation

    # Sprawdź NDVI
    ndvi_val = ndvi[row, col]
    ndvi_range = max(ndvi_max - ndvi_min, 0.1) * range_multiplier
    if ndvi_val < ndvi_min:
        total_deviation += (ndvi_min - ndvi_val) / ndvi_range
    elif ndvi_val > ndvi_max:
        total_deviation += (ndvi_val - ndvi_max) / ndvi_range

    return total_deviation


def measure_width_at_point(point: tuple, direction: tuple, cluster_cells_set: set,
                           max_width: int = 30) -> int:
    """
    Mierzy szerokość klastra w punkcie prostopadle do kierunku ścieżki.

    Args:
        point: Punkt (row, col)
        direction: Kierunek ścieżki (dy, dx)
        cluster_cells_set: Zbiór komórek klastra
        max_width: Maksymalna szerokość do sprawdzenia

    Returns:
        Szerokość w pikselach
    """
    row, col = point
    dy, dx = direction

    # Wektor prostopadły
    perp_dy, perp_dx = -dx, dy

    # Normalizuj
    length = math.sqrt(perp_dy**2 + perp_dx**2)
    if length < 1e-6:
        return 1
    perp_dy /= length
    perp_dx /= length

    # Szukaj w obie strony prostopadle do kierunku
    width = 1  # Sam punkt

    for direction_mult in [1, -1]:
        for dist in range(1, max_width):
            nr = int(round(row + direction_mult * dist * perp_dy))
            nc = int(round(col + direction_mult * dist * perp_dx))

            if (nr, nc) in cluster_cells_set:
                width += 1
            else:
                break

    return width


def analyze_path_width(path: list, cluster_cells: list, sample_step: int = 10,
                       direction_sample: int = 5) -> tuple:
    """
    Analizuje szerokość klastra wzdłuż ścieżki.

    Args:
        path: Lista punktów ścieżki
        cluster_cells: Lista komórek klastra
        sample_step: Co ile pikseli próbkować
        direction_sample: Liczba pikseli do obliczenia kierunku

    Returns:
        Tuple (średnia_szerokość, wariancja_szerokości, lista_szerokości)
    """
    if len(path) < direction_sample * 2 + 1:
        return 1.0, 0.0, [1]

    cluster_cells_set = set(cluster_cells)
    widths = []

    for i in range(direction_sample, len(path) - direction_sample, sample_step):
        p_prev = path[i - direction_sample]
        p_next = path[i + direction_sample]

        dy = p_next[0] - p_prev[0]
        dx = p_next[1] - p_prev[1]

        length = math.sqrt(dy**2 + dx**2)
        if length < 1e-6:
            continue

        direction = (dy / length, dx / length)
        width = measure_width_at_point(path[i], direction, cluster_cells_set)
        widths.append(width)

    if not widths:
        return 1.0, 0.0, [1]

    avg_width = sum(widths) / len(widths)
    variance = sum((w - avg_width)**2 for w in widths) / len(widths)

    return avg_width, variance, widths


def is_valid_railway_width(cluster, min_width: int = 2, max_width: int = 8,
                           max_variance: float = 3.0, min_valid_ratio: float = 0.7) -> tuple:
    """
    Sprawdza czy klaster ma szerokość charakterystyczną dla torów kolejowych.

    Args:
        cluster: Obiekt klastra
        min_width: Minimalna szerokość
        max_width: Maksymalna szerokość
        max_variance: Maksymalna wariancja
        min_valid_ratio: Minimalny stosunek prawidłowych próbek

    Returns:
        Tuple (czy_prawidłowy, średnia_szerokość, wariancja, powód_odrzucenia)
    """
    path = cluster.longest_path
    if path is None or len(path) < 20:
        return True, 0, 0, None

    avg_width, variance, widths = analyze_path_width(path, cluster.cells)

    # Sprawdź ile próbek ma prawidłową szerokość
    valid_samples = sum(1 for w in widths if min_width <= w <= max_width)
    valid_ratio = valid_samples / len(widths) if widths else 0

    # Odrzuć jeśli za szeroki (autostrada)
    if avg_width > max_width:
        return False, avg_width, variance, f"za_szeroki ({avg_width:.1f} > {max_width})"

    # Odrzuć jeśli za duża wariancja
    if variance > max_variance and avg_width > 4:
        return False, avg_width, variance, f"niestabilna_szerokosc (var={variance:.1f})"

    # Odrzuć jeśli za mało próbek ma prawidłową szerokość
    if valid_ratio < min_valid_ratio:
        return False, avg_width, variance, f"za_malo_prawidlowych ({valid_ratio:.1%})"

    return True, avg_width, variance, None


def calculate_cluster_thickness(cluster) -> float:
    """
    Oblicza średnią grubość klastra jako stosunek powierzchni do długości.

    Args:
        cluster: Obiekt klastra

    Returns:
        Grubość klastra
    """
    if cluster.depth <= 0:
        return float('inf')

    thickness = cluster.size() / cluster.depth
    return thickness


def check_parallel_false_positives(clusters_list: list, min_distance: int = 20,
                                   max_angle_diff: int = 15,
                                   min_depth_threshold: int = 130) -> set:
    """
    Wykrywa pary równoległych klastrów które są blisko siebie.

    Args:
        clusters_list: Lista klastrów
        min_distance: Minimalna odległość między klastrami
        max_angle_diff: Maksymalna różnica kąta dla równoległości
        min_depth_threshold: Minimalna głębokość do analizy

    Returns:
        Zbiór id klastrów do odrzucenia
    """
    clusters_to_reject = set()

    valid_clusters = [c for c in clusters_list if c.depth >= min_depth_threshold
                      and c.longest_path is not None and len(c.longest_path) >= 20]

    for i, c1 in enumerate(valid_clusters):
        if id(c1) in clusters_to_reject:
            continue

        path1 = c1.longest_path
        dir1 = get_direction_vector(path1, from_end=True, sample_length=min(50, len(path1)//2))

        for c2 in valid_clusters[i+1:]:
            if id(c2) in clusters_to_reject:
                continue

            path2 = c2.longest_path
            dir2 = get_direction_vector(path2, from_end=True, sample_length=min(50, len(path2)//2))

            # Sprawdź czy kierunki są równoległe
            dot = abs(dir1[0]*dir2[0] + dir1[1]*dir2[1])
            if dot < math.cos(math.radians(max_angle_diff)):
                continue

            # Sprawdź odległość między klastrami
            min_dist = float('inf')
            sample_indices1 = [0, len(path1)//4, len(path1)//2, 3*len(path1)//4, len(path1)-1]
            sample_indices2 = [0, len(path2)//4, len(path2)//2, 3*len(path2)//4, len(path2)-1]
            for idx1 in sample_indices1:
                if idx1 >= len(path1):
                    continue
                p1 = path1[idx1]
                for idx2 in sample_indices2:
                    if idx2 >= len(path2):
                        continue
                    p2 = path2[idx2]
                    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    min_dist = min(min_dist, dist)

            if min_dist < min_distance:
                if c1.depth < c2.depth:
                    clusters_to_reject.add(id(c1))
                else:
                    clusters_to_reject.add(id(c2))

    return clusters_to_reject

