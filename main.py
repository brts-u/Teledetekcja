"""
Główny skrypt do wykrywania torów kolejowych z obrazu satelitarnego.

Uruchomienie:
    python main.py

Konfiguracja:
    Wszystkie parametry znajdują się w pliku config.py
"""

import os
import numpy as np
import rasterio

# Import konfiguracji
import config

# Import modułów projektu
from io_utils import load_raster_bands, safe_rasterio_write
from mask_operations import create_confidence_mask, extend_all_track_endpoints
from clusters import label_with_diagonals, create_clusters
from vectorization import TrackVectorizer
from json_scraper import arr


def create_initial_mask(all_bands, ndvi, transform, crs):
    """
    Tworzy początkową maskę spektralną i rozszerza ją od końców torów.

    Returns:
        Maska wynikowa
    """
    # KROK 1: Utwórz ścisłą maskę (wysoką pewność - "core" torów)
    print("Creating strict boolean mask (high confidence cores)...")

    # Rozdziel pasma dla funkcji create_confidence_mask
    bands_list = [all_bands[i] for i in range(8)]

    strict_mask, strict_confidence = create_confidence_mask(
        bands_list, arr,
        config.STRICT_SCALE_MIN, config.STRICT_SCALE_MAX,
        ndvi, config.NDVI_MIN_STRICT, config.NDVI_MAX_STRICT
    )

    strict_count = np.sum(strict_mask)
    print(f"  Ścisła maska: {strict_count} pikseli")

    # Zapisz ścisłą maskę do podglądu
    safe_rasterio_write('strict_mask.tif', strict_mask.astype('uint8'),
                        strict_mask.shape[0], strict_mask.shape[1], crs, transform)

    # KROK 2: Wstępna klasteryzacja i analiza kątów
    print("Performing initial clustering and angle analysis...")
    labeled_initial, num_initial = label_with_diagonals(strict_mask)
    print(f"  Znaleziono {num_initial} wstępnych klastrów")

    initial_clusters = create_clusters(labeled_initial)
    print(f"  Utworzono {len(initial_clusters)} klastrów")

    # Oblicz głębokość dla klastrów >= 50 pikseli
    print("  Obliczanie głębokości klastrów...")
    for cluster in initial_clusters.values():
        if cluster.size() >= 50:
            cluster.get_depth()

    # KROK 3: Rozszerzanie od końców prawidłowych torów
    print("Extending tracks from valid endpoints (no sharp angles)...")
    result_mask = extend_all_track_endpoints(
        strict_mask,
        list(initial_clusters.values()),
        all_bands, arr,
        config.LOOSE_SCALE_MIN, config.LOOSE_SCALE_MAX,
        ndvi, config.NDVI_MIN_LOOSE, config.NDVI_MAX_LOOSE,
        min_depth=config.MIN_DEPTH_FOR_VECTORIZATION,
        max_distance=config.EXTENSION_MAX_DISTANCE,
        min_bands_match=config.EXTENSION_MIN_BANDS,
        min_angle=config.MIN_ANGLE_DEGREES
    )

    extended_count = np.sum(result_mask)
    print(f"  Po rozszerzeniu: {extended_count} pikseli (+{extended_count - strict_count})")

    # KROK 4: Ponowna klasteryzacja i drugie rozszerzenie
    print("Second pass: re-clustering and extending...")
    labeled_second, num_second = label_with_diagonals(result_mask)
    second_clusters = create_clusters(labeled_second)

    for cluster in second_clusters.values():
        if cluster.size() >= 50:
            cluster.get_depth()

    result_mask = extend_all_track_endpoints(
        result_mask,
        list(second_clusters.values()),
        all_bands, arr,
        config.LOOSE_SCALE_MIN, config.LOOSE_SCALE_MAX,
        ndvi, config.NDVI_MIN_LOOSE, config.NDVI_MAX_LOOSE,
        min_depth=100,
        max_distance=config.EXTENSION_MAX_DISTANCE,
        min_bands_match=config.EXTENSION_MIN_BANDS + 1,
        min_angle=config.MIN_ANGLE_DEGREES
    )

    final_count = np.sum(result_mask)
    print(f"  Finalna maska: {final_count} pikseli (+{final_count - strict_count} od początku)")

    # Zapisz wynikowy raster
    safe_rasterio_write('result_mask.tif', result_mask.astype('uint8'),
                        result_mask.shape[0], result_mask.shape[1], crs, transform)

    # Zapisz mapę pewności
    safe_rasterio_write('confidence_map.tif', strict_confidence,
                        strict_confidence.shape[0], strict_confidence.shape[1],
                        crs, transform, dtype='int32')

    return result_mask


def main():
    """Główna funkcja programu."""

    print("="*60)
    print("WYKRYWANIE TORÓW KOLEJOWYCH")
    print("="*60)

    # Wczytaj dane rastrowe
    all_bands, ndvi, transform, crs = load_raster_bands(config.INPUT_RASTER_PATH)

    # Sprawdź czy maska już istnieje
    result_mask = None
    if not config.SKIP_BOOLEAN_MASK or not os.path.exists('result_mask.tif'):
        result_mask = create_initial_mask(all_bands, ndvi, transform, crs)
    else:
        print("Using existing boolean mask...")
        with rasterio.open('result_mask.tif') as src:
            result_mask = src.read(1).astype(bool)

    # Inicjalizuj wektoryzator
    vectorizer = TrackVectorizer(transform, crs, all_bands, ndvi, arr)

    # Wczytaj i przetworz maskę
    mask = vectorizer.load_and_preprocess_mask('result_mask.tif')

    # Utwórz klastry z głębokościami
    clusters, labeled_img = vectorizer.create_clusters_with_depth(mask)

    # Zapisz raster głębokości
    vectorizer.save_depth_raster(clusters, labeled_img)

    # Pierwsza wektoryzacja
    print("\n" + "="*60)
    print("PIERWSZA WEKTORYZACJA")
    print("="*60)
    gdf_raw, stats = vectorizer.filter_and_vectorize(clusters, 'train_tracks_raw.geojson')

    # Łączenie segmentów
    new_mask, connections = vectorizer.connect_segments(clusters, result_mask)

    # Finalna wektoryzacja
    final_gdf = vectorizer.final_vectorization(new_mask, 'train_tracks.geojson')

    print("\n" + "="*60)
    print("ZAKOŃCZONO")
    print("="*60)
    print(f"Wyniki zapisano do: train_tracks.geojson")
    print(f"Znaleziono {len(final_gdf)} tras kolejowych")


if __name__ == "__main__":
    main()
