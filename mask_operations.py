"""
Operacje na maskach - tworzenie masek spektralnych, rozszerzanie, rozszerzanie końców torów.
"""

import math
import numpy as np
from scipy.ndimage import binary_dilation

from clusters import label_with_diagonals
from path_analysis import (
    check_pixel_threshold,
    find_cluster_endpoints_with_direction,
    check_path_angles
)


def create_confidence_mask(bands: list, arr: np.ndarray, scale_min: float, scale_max: float,
                           ndvi: np.ndarray, ndvi_min: float, ndvi_max: float) -> tuple:
    """
    Tworzy maskę na podstawie progów pewności dla wszystkich pasm.

    Args:
        bands: Lista tablic numpy z pasmami spektralnymi
        arr: Tablica ze statystykami (min, mean, max) dla każdego pasma
        scale_min: Mnożnik dla minimalnych wartości
        scale_max: Mnożnik dla maksymalnych wartości
        ndvi: Tablica z wartościami NDVI
        ndvi_min: Minimalny próg NDVI
        ndvi_max: Maksymalny próg NDVI

    Returns:
        Tuple (result_mask, confidence_map):
        - result_mask: Maska boolean (piksele spełniające wszystkie kryteria)
        - confidence_map: Mapa pewności (ile pasm pasuje dla każdego piksela)
    """
    height, width = bands[0].shape
    result_mask = np.ones((height, width), dtype=bool)
    confidence_map = np.zeros((height, width), dtype=np.int32)

    for i, (band, (min_val, mean_val, max_val)) in enumerate(zip(bands, arr)):
        min_thresh = min_val * scale_min
        max_thresh = max_val * scale_max
        band_mask = (band >= min_thresh) & (band <= max_thresh)
        result_mask &= band_mask
        confidence_map += band_mask.astype(np.int32)

    # Dodaj warunek NDVI
    ndvi_mask = (ndvi >= ndvi_min) & (ndvi <= ndvi_max)
    result_mask &= ndvi_mask
    confidence_map += ndvi_mask.astype(np.int32)

    return result_mask, confidence_map


def expand_mask_from_endpoints(mask: np.ndarray, bands: np.ndarray, arr: np.ndarray,
                               scale_min: float, scale_max: float,
                               ndvi: np.ndarray, ndvi_min: float, ndvi_max: float,
                               min_cluster_size: int = 50, max_iterations: int = 100,
                               min_bands_match: int = 6) -> np.ndarray:
    """
    Rozszerza maskę poprzez sprawdzanie sąsiadów na krawędziach dużych klastrów.

    Args:
        mask: Maska wejściowa
        bands: Tablica z pasmami spektralnymi (8, H, W)
        arr: Statystyki spektralne
        scale_min/scale_max: Mnożniki progów
        ndvi: Tablica NDVI
        ndvi_min/ndvi_max: Progi NDVI
        min_cluster_size: Minimalny rozmiar klastra do rozszerzania
        max_iterations: Maksymalna liczba iteracji
        min_bands_match: Minimalna liczba pasujących pasm

    Returns:
        Rozszerzona maska
    """
    expanded_mask = mask.copy()
    height, width = mask.shape

    # Znajdź duże klastry do rozszerzania
    labeled_img, num_features = label_with_diagonals(mask)
    cluster_sizes = {}
    for label_id in range(1, num_features + 1):
        cluster_sizes[label_id] = np.sum(labeled_img == label_id)

    # Maska tylko dużych klastrów
    large_cluster_mask = np.zeros_like(mask)
    for label_id, size in cluster_sizes.items():
        if size >= min_cluster_size:
            large_cluster_mask |= (labeled_img == label_id)

    large_cluster_count = np.sum(large_cluster_mask)
    num_large = sum(1 for s in cluster_sizes.values() if s >= min_cluster_size)
    print(f"  Rozszerzanie od {num_large} dużych klastrów ({large_cluster_count} pikseli)")

    total_added = 0

    for iteration in range(max_iterations):
        # Znajdź piksele na krawędziach (tylko 1 piksel od maski)
        dilated = binary_dilation(expanded_mask & large_cluster_mask)
        candidates = dilated & ~expanded_mask

        candidate_coords = np.argwhere(candidates)

        if len(candidate_coords) == 0:
            print(f"  Zatrzymano - brak kandydatów")
            break

        added_this_iteration = 0
        new_pixels = []

        for nr, nc in candidate_coords:
            if check_pixel_threshold(nr, nc, bands, arr, scale_min, scale_max,
                                     ndvi, ndvi_min, ndvi_max, min_bands_match):
                new_pixels.append((nr, nc))
                added_this_iteration += 1

        # Dodaj nowe piksele do maski
        for nr, nc in new_pixels:
            expanded_mask[nr, nc] = True
            large_cluster_mask[nr, nc] = True

        total_added += added_this_iteration
        print(f"  Iteracja {iteration + 1}: dodano {added_this_iteration} pikseli")

        if added_this_iteration == 0:
            print(f"  Zatrzymano - brak nowych pikseli do dodania")
            break

    print(f"Łącznie dodano {total_added} pikseli przez rozszerzanie")
    return expanded_mask


def extend_track_in_direction(start_point: tuple, direction: tuple, mask: np.ndarray,
                              bands: np.ndarray, arr: np.ndarray, scale_min: float, scale_max: float,
                              ndvi: np.ndarray, ndvi_min: float, ndvi_max: float,
                              max_distance: int = 0, min_bands_match: int = 7,
                              search_radius: int = 3, min_angle: int = 120) -> list:
    """
    Rozszerza tor w danym kierunku, szukając pikseli spełniających kryteria
    i utrzymując płynny kąt (bez ostrych zakrętów).

    Args:
        start_point: Punkt startowy (row, col)
        direction: Wektor kierunku (dr, dc)
        mask: Aktualna maska
        bands: Pasma spektralne
        arr: Statystyki spektralne
        scale_min/scale_max: Mnożniki progów
        ndvi: NDVI
        ndvi_min/ndvi_max: Progi NDVI
        max_distance: Maksymalna odległość (0 = bez limitu)
        min_bands_match: Minimalna liczba pasujących pasm
        search_radius: Promień szukania
        min_angle: Minimalny kąt

    Returns:
        Lista nowych pikseli do dodania
    """
    height, width = mask.shape
    new_pixels = []
    new_pixels_set = set()

    current_pos = start_point
    current_dir = direction

    # Jeśli max_distance=0, ustaw na maksymalny wymiar rastra
    effective_max_distance = max_distance if max_distance > 0 else max(height, width)

    for step in range(effective_max_distance):
        best_pixel = None
        best_score = -float('inf')

        # Sprawdź piksele w stożku w kierunku current_dir
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                if dr == 0 and dc == 0:
                    continue

                nr = int(current_pos[0] + dr)
                nc = int(current_pos[1] + dc)

                # Sprawdź granice
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                # Sprawdź czy już w masce lub nowych pikselach
                if mask[nr, nc] or (nr, nc) in new_pixels_set:
                    continue

                # Oblicz kąt między aktualnym kierunkiem a nowym
                move_mag = math.sqrt(dr**2 + dc**2)
                if move_mag == 0:
                    continue
                move_dir_norm = (dr / move_mag, dc / move_mag)

                # Iloczyn skalarny daje cosinus kąta
                dot = current_dir[0] * move_dir_norm[0] + current_dir[1] * move_dir_norm[1]

                # Preferuj piksele w kierunku ruchu (dot bliski 1)
                if dot < 0.5:  # Odrzuć piksele pod kątem > 60 stopni
                    continue

                # Sprawdź czy piksel spełnia próg spektralny
                if not check_pixel_threshold(nr, nc, bands, arr, scale_min, scale_max,
                                             ndvi, ndvi_min, ndvi_max, min_bands_match):
                    continue

                score = dot

                if score > best_score:
                    best_score = score
                    best_pixel = (nr, nc)

        if best_pixel is None:
            break

        new_pixels.append(best_pixel)
        new_pixels_set.add(best_pixel)

        # Aktualizuj kierunek (uśrednij z poprzednim dla płynności)
        dr = best_pixel[0] - current_pos[0]
        dc = best_pixel[1] - current_pos[1]
        mag = math.sqrt(dr**2 + dc**2)
        if mag > 0:
            new_dir = (dr / mag, dc / mag)
            # Płynna aktualizacja kierunku (70% nowy, 30% stary)
            current_dir = (0.7 * new_dir[0] + 0.3 * current_dir[0],
                           0.7 * new_dir[1] + 0.3 * current_dir[1])
            # Renormalizuj
            mag = math.sqrt(current_dir[0]**2 + current_dir[1]**2)
            if mag > 0:
                current_dir = (current_dir[0] / mag, current_dir[1] / mag)

        current_pos = best_pixel

    return new_pixels


def extend_all_track_endpoints(mask: np.ndarray, clusters: list, bands: np.ndarray,
                               arr: np.ndarray, scale_min: float, scale_max: float,
                               ndvi: np.ndarray, ndvi_min: float, ndvi_max: float,
                               min_depth: int = 100, max_distance: int = 100,
                               min_bands_match: int = 7, min_angle: int = 120) -> np.ndarray:
    """
    Rozszerza wszystkie prawidłowe tory od ich punktów końcowych.
    Pracuje tylko na dużych klastrach bez ostrych kątów.

    Args:
        mask: Maska wejściowa
        clusters: Lista klastrów
        bands: Pasma spektralne
        arr: Statystyki spektralne
        scale_min/scale_max: Mnożniki progów
        ndvi: NDVI
        ndvi_min/ndvi_max: Progi NDVI
        min_depth: Minimalna głębokość klastra
        max_distance: Maksymalna odległość rozszerzania
        min_bands_match: Minimalna liczba pasujących pasm
        min_angle: Minimalny kąt

    Returns:
        Rozszerzona maska
    """
    extended_mask = mask.copy()
    total_added = 0
    tracks_extended = 0

    print(f"  Analizowanie {len(clusters)} klastrów...")

    for cluster in clusters:
        if cluster.depth < min_depth:
            continue

        # Znajdź punkty końcowe z kierunkami
        endpoints = find_cluster_endpoints_with_direction(cluster, min_angle=min_angle)

        for endpoint, direction, is_valid in endpoints:
            if not is_valid:
                continue

            if direction == (0, 0):
                continue

            # Rozszerz w tym kierunku
            new_pixels = extend_track_in_direction(
                endpoint, direction, extended_mask, bands, arr,
                scale_min, scale_max, ndvi, ndvi_min, ndvi_max,
                max_distance=max_distance, min_bands_match=min_bands_match,
                min_angle=min_angle
            )

            if new_pixels:
                for r, c in new_pixels:
                    extended_mask[r, c] = True
                total_added += len(new_pixels)
                tracks_extended += 1

    print(f"  Rozszerzono {tracks_extended} końców torów, dodano {total_added} pikseli")
    return extended_mask

