"""
Wektoryzacja klastrów do formatu GeoJSON.
"""

import sys
import time
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString
from skimage.morphology import closing, opening, square

from clusters import label_with_diagonals, create_clusters
from path_analysis import (
    calculate_cluster_thickness,
    is_valid_railway_width,
    check_parallel_false_positives,
    check_path_has_sharp_turns,
    get_direction_vector
)
from path_connection import (
    find_vectors_in_cone,
    connect_clusters_with_pixels,
    check_connection_angle
)
from io_utils import safe_rasterio_write, safe_geojson_write
import config


class TrackVectorizer:
    """
    Klasa do wektoryzacji klastrów torów kolejowych.
    """

    def __init__(self, transform, crs, all_bands, ndvi, arr):
        """
        Inicjalizuje wektoryzator.

        Args:
            transform: Transformacja geoprzestrzenna
            crs: Układ współrzędnych
            all_bands: Pasma spektralne
            ndvi: NDVI
            arr: Statystyki spektralne
        """
        self.transform = transform
        self.crs = crs
        self.all_bands = all_bands
        self.ndvi = ndvi
        self.arr = arr

    def load_and_preprocess_mask(self, mask_path: str) -> np.ndarray:
        """
        Wczytuje i przetwarza maskę morfologicznie.

        Args:
            mask_path: Ścieżka do pliku maski

        Returns:
            Przetworzona maska
        """
        import rasterio

        print("Performing morphological operations...")
        with rasterio.open(mask_path) as src:
            img = src.read(1).astype(bool)

        kernel3 = square(3)
        closed_img = closing(img, kernel3)

        kernel2 = square(2)
        opened_img = opening(closed_img, kernel2)

        return img

    def create_clusters_with_depth(self, mask: np.ndarray) -> list:
        """
        Tworzy klastry i oblicza ich głębokości.

        Args:
            mask: Maska binarna

        Returns:
            Lista klastrów posortowana wg głębokości
        """
        print("Creating clusters...")
        t0 = time.time()

        labeled_img, num_features = label_with_diagonals(mask)
        clusters = create_clusters(labeled_img)

        t1 = time.time()
        print(f"Total clusters found: {len(clusters)}, {t1 - t0:.2f} seconds")

        # Oblicz głębokości
        last_percent = 0
        size_sum = sum(c.size() for c in clusters.values())

        print('Calculating depths for clusters (this may take a while)...')
        partial_size_sum = 0
        t0 = time.time()

        for cluster in clusters.values():
            if cluster.size() >= 100:
                cluster.get_depth()
            partial_size_sum += cluster.size()
            percent = (partial_size_sum + 1) * 100 // size_sum
            if percent > last_percent:
                elapsed = time.time() - t0
                sys.stdout.write(
                    f"\rProgress: {percent}% (est. time remaining {elapsed * (100 - percent) / (percent + 1):.2f} seconds)")
                sys.stdout.flush()
                last_percent = percent

        sys.stdout.write("\nDone!\n")

        clusters_list = list(clusters.values())
        clusters_list.sort(key=lambda c: c.depth, reverse=True)

        return clusters_list, labeled_img

    def save_depth_raster(self, clusters: list, labeled_img: np.ndarray) -> None:
        """
        Zapisuje raster głębokości.

        Args:
            clusters: Lista klastrów
            labeled_img: Obraz z etykietami
        """
        depth_raster = np.zeros(labeled_img.shape, dtype='float32')
        for cluster in clusters:
            depth = cluster.depth
            for (row, col) in cluster.cells:
                depth_raster[row, col] = depth

        safe_rasterio_write('depths.tif', depth_raster,
                            labeled_img.shape[0], labeled_img.shape[1],
                            self.crs, self.transform, count=1, dtype='float32')

    def filter_and_vectorize(self, clusters: list, output_path: str) -> tuple:
        """
        Filtruje klastry i tworzy wektory.

        Args:
            clusters: Lista klastrów
            output_path: Ścieżka do pliku wyjściowego

        Returns:
            Tuple (GeoDataFrame, statystyki)
        """
        print(f"Filtering clusters: depth >= {config.MIN_DEPTH_FOR_VECTORIZATION}, "
              f"thickness in [{config.MIN_CLUSTER_THICKNESS}, {config.MAX_CLUSTER_THICKNESS}]")
        print(f"Width filter: {config.EXPECTED_TRACK_WIDTH_MIN}-{config.EXPECTED_TRACK_WIDTH_MAX} px, "
              f"max variance {config.WIDTH_VARIANCE_THRESHOLD}")

        # Wykryj równoległe fałszywe linie
        print("Checking for parallel false positives...")
        parallel_rejects = check_parallel_false_positives(
            clusters,
            min_distance=config.MIN_DISTANCE_BETWEEN_TRACKS,
            max_angle_diff=config.PARALLEL_ANGLE_THRESHOLD,
            min_depth_threshold=config.MIN_DEPTH_FOR_VECTORIZATION
        )
        print(f"  Oznaczono {len(parallel_rejects)} klastrów jako potencjalne fałszywe (równoległe)")

        data = {'geometry': [], 'depth': [], 'thickness': [], 'avg_width': [], 'width_variance': []}
        stats = {
            'skipped_thin': 0,
            'skipped_thick': 0,
            'skipped_short': 0,
            'skipped_width': 0,
            'skipped_parallel': 0
        }

        for cluster in clusters:
            if cluster.depth < config.MIN_DEPTH_FOR_VECTORIZATION:
                stats['skipped_short'] += 1
                continue

            if id(cluster) in parallel_rejects:
                stats['skipped_parallel'] += 1
                continue

            thickness = calculate_cluster_thickness(cluster)

            if thickness < config.MIN_CLUSTER_THICKNESS:
                stats['skipped_thin'] += 1
                continue

            if thickness > config.MAX_CLUSTER_THICKNESS:
                stats['skipped_thick'] += 1
                continue

            is_valid_width, avg_width, width_var, reject_reason = is_valid_railway_width(
                cluster,
                min_width=config.EXPECTED_TRACK_WIDTH_MIN,
                max_width=config.EXPECTED_TRACK_WIDTH_MAX,
                max_variance=config.WIDTH_VARIANCE_THRESHOLD,
                min_valid_ratio=config.MIN_VALID_WIDTH_RATIO
            )

            if not is_valid_width:
                stats['skipped_width'] += 1
                print(f"    Odrzucono klaster (depth={cluster.depth}): {reject_reason}")
                continue

            # Konwertuj ścieżkę do geometrii
            full_path = cluster.longest_path
            steps = full_path[::10]
            geo_coords = []
            for row, col in steps:
                x, y = self.transform * (col, row)
                geo_coords.append((x, y))

            line = LineString(geo_coords)
            data['geometry'].append(line)
            data['depth'].append(cluster.depth)
            data['thickness'].append(thickness)
            data['avg_width'].append(avg_width)
            data['width_variance'].append(width_var)

        print(f"  Pominięto: {stats['skipped_short']} za krótkich, "
              f"{stats['skipped_thin']} za cienkich, {stats['skipped_thick']} za grubych")
        print(f"  Pominięto: {stats['skipped_width']} nieprawidłowa szerokość, "
              f"{stats['skipped_parallel']} równoległe fałszywe")
        print(f"  Zachowano: {len(data['geometry'])} klastrów")

        gdf = GeoDataFrame(data, crs=self.crs)
        safe_geojson_write(gdf, output_path)

        return gdf, stats

    def connect_segments(self, clusters: list, result_mask: np.ndarray) -> tuple:
        """
        Łączy odcinki torów.

        Args:
            clusters: Lista klastrów
            result_mask: Maska wynikowa

        Returns:
            Tuple (nowa_maska, liczba_połączeń)
        """
        print("\n" + "="*60)
        print("DRUGA WEKTORYZACJA - Łączenie odcinków i filtrowanie")
        print("="*60)

        # Zbierz prawidłowe klastry do łączenia
        valid_clusters = []
        for cluster in clusters:
            if cluster.depth < config.MIN_DEPTH_FOR_VECTORIZATION:
                continue
            thickness = calculate_cluster_thickness(cluster)
            if thickness < config.MIN_CLUSTER_THICKNESS or thickness > config.MAX_CLUSTER_THICKNESS:
                continue
            is_valid_width, avg_width, width_var, _ = is_valid_railway_width(
                cluster,
                min_width=config.EXPECTED_TRACK_WIDTH_MIN,
                max_width=config.EXPECTED_TRACK_WIDTH_MAX,
                max_variance=config.WIDTH_VARIANCE_THRESHOLD,
                min_valid_ratio=config.MIN_VALID_WIDTH_RATIO
            )
            if is_valid_width:
                valid_clusters.append(cluster)

        print(f"Klastrów do łączenia: {len(valid_clusters)}")

        # Łączenie klastrów
        print("Szukanie połączeń między odcinkami...")

        connections_made = 0
        connected_pairs = set()
        new_mask = result_mask.copy()

        for cluster in valid_clusters:
            path = cluster.longest_path
            if path is None or len(path) < 20:
                continue

            for end_idx in [0, -1]:
                endpoint = path[end_idx]

                sample_len = min(20, len(path)//4)
                if end_idx == 0:
                    direction = get_direction_vector(path, from_end=False, sample_length=sample_len)
                else:
                    direction = get_direction_vector(path, from_end=True, sample_length=sample_len)

                if direction == (0, 0):
                    continue

                candidates = find_vectors_in_cone(
                    endpoint, direction, valid_clusters, cluster,
                    cone_angle=config.VECTOR_CONNECT_CONE_ANGLE,
                    max_distance=config.VECTOR_CONNECT_MAX_DISTANCE
                )

                for other_cluster, other_end, dist, angle, other_end_idx in candidates:
                    pair_key = tuple(sorted([id(cluster), id(other_cluster)]))
                    if pair_key in connected_pairs:
                        continue

                    angle_ok, connection_angle = check_connection_angle(
                        cluster, end_idx, other_cluster, other_end_idx,
                        max_angle=config.MAX_SHARP_ANGLE_AFTER_CONNECT
                    )

                    if not angle_ok:
                        continue

                    connecting_pixels = connect_clusters_with_pixels(
                        endpoint, other_end, new_mask, self.all_bands, self.arr, self.ndvi,
                        config.NDVI_MIN_LOOSE, config.NDVI_MAX_LOOSE,
                        min_bands_match=config.VECTOR_CONNECT_MIN_BANDS
                    )

                    if connecting_pixels is not None:
                        for r, c in connecting_pixels:
                            new_mask[r, c] = True

                        connected_pairs.add(pair_key)
                        connections_made += 1
                        print(f"  Połączono: depth {cluster.depth} <-> {other_cluster.depth}, "
                              f"dist={dist:.0f}, angle={connection_angle:.1f}°, "
                              f"nowe piksele={len(connecting_pixels)}")
                        break

        print(f"Utworzono {connections_made} połączeń")

        safe_rasterio_write('result_mask_connected.tif', new_mask.astype('uint8'),
                            new_mask.shape[0], new_mask.shape[1], self.crs, self.transform)

        return new_mask, connections_made

    def final_vectorization(self, mask: np.ndarray, output_path: str) -> GeoDataFrame:
        """
        Końcowa wektoryzacja z filtrowaniem zakrętów.

        Args:
            mask: Maska binarna
            output_path: Ścieżka do pliku wyjściowego

        Returns:
            GeoDataFrame z wynikami
        """
        print("\n" + "="*60)
        print("TRZECIA WEKTORYZACJA - Finalna z filtrowaniem zakrętów")
        print("="*60)

        labeled_final, num_final = label_with_diagonals(mask)
        final_clusters = create_clusters(labeled_final)
        print(f"Klastrów po połączeniu: {len(final_clusters)}")

        for cluster in final_clusters.values():
            if cluster.size() >= 50:
                cluster.get_depth()

        final_data = {'geometry': [], 'depth': [], 'thickness': [], 'avg_width': []}
        skipped_sharp_turns = 0
        skipped_highway = 0

        final_clusters_list = list(final_clusters.values())
        final_clusters_list.sort(key=lambda c: c.depth, reverse=True)

        for cluster in final_clusters_list:
            if cluster.depth < config.MIN_DEPTH_FOR_VECTORIZATION:
                continue

            thickness = calculate_cluster_thickness(cluster)
            if thickness < config.MIN_CLUSTER_THICKNESS or thickness > config.MAX_CLUSTER_THICKNESS:
                continue

            is_valid_width, avg_width, width_var, reject_reason = is_valid_railway_width(
                cluster,
                min_width=config.EXPECTED_TRACK_WIDTH_MIN,
                max_width=config.EXPECTED_TRACK_WIDTH_MAX,
                max_variance=config.WIDTH_VARIANCE_THRESHOLD,
                min_valid_ratio=config.MIN_VALID_WIDTH_RATIO
            )

            if not is_valid_width:
                if (avg_width >= config.HIGHWAY_MIN_WIDTH and
                    width_var <= config.HIGHWAY_LOW_VARIANCE and
                    cluster.depth >= config.HIGHWAY_MIN_LENGTH):
                    skipped_highway += 1
                    print(f"  Odrzucono autostradę: depth={cluster.depth}, width={avg_width:.1f}, var={width_var:.2f}")
                continue

            # Sprawdź ostre zakręty
            path = cluster.longest_path
            if path is not None and len(path) >= 30:
                has_sharp, angles = check_path_has_sharp_turns(path, max_angle=config.MAX_SHARP_ANGLE_AFTER_CONNECT)
                if has_sharp:
                    min_angle = min(angles) if angles else 180
                    skipped_sharp_turns += 1
                    print(f"  Odrzucono (ostry zakręt): depth={cluster.depth}, min_angle={min_angle:.1f}°")
                    continue

            full_path = cluster.longest_path
            steps = full_path[::10]
            geo_coords = []
            for row, col in steps:
                x, y = self.transform * (col, row)
                geo_coords.append((x, y))

            line = LineString(geo_coords)
            final_data['geometry'].append(line)
            final_data['depth'].append(cluster.depth)
            final_data['thickness'].append(thickness)
            final_data['avg_width'].append(avg_width)

        print(f"\nStatystyki końcowe:")
        print(f"  Odrzucono: {skipped_sharp_turns} z ostrymi zakrętami, {skipped_highway} autostrad")
        print(f"  Zachowano: {len(final_data['geometry'])} tras kolejowych")

        final_gdf = GeoDataFrame(final_data, crs=self.crs)
        safe_geojson_write(final_gdf, output_path)

        return final_gdf

