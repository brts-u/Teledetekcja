"""
Narzędzia do operacji wejścia/wyjścia - zapis i odczyt plików.
"""

import os
import datetime
import rasterio
import numpy as np


def safe_rasterio_write(filepath: str, data: np.ndarray, height: int, width: int,
                        crs, transform, count: int = 1, dtype: str = 'uint8') -> str:
    """
    Bezpiecznie zapisuje plik rasterowy. Jeśli plik jest zablokowany,
    próbuje zapisać pod alternatywną nazwą.

    Args:
        filepath: Ścieżka do pliku wyjściowego
        data: Dane do zapisania
        height: Wysokość rastra
        width: Szerokość rastra
        crs: Układ współrzędnych
        transform: Transformacja geoprzestrzenna
        count: Liczba pasm
        dtype: Typ danych

    Returns:
        Ścieżka do zapisanego pliku
    """
    try:
        with rasterio.open(filepath, 'w', driver='GTiff', height=height, width=width,
                           crs=crs, transform=transform, count=count, dtype=dtype) as dst:
            dst.write(data, 1)
        print(f"  Zapisano: {filepath}")
        return filepath
    except Exception as e:
        # Plik zablokowany - użyj alternatywnej nazwy z timestampem
        base, ext = os.path.splitext(filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_filepath = f"{base}_{timestamp}{ext}"
        print(f"  Uwaga: Nie można zapisać {filepath} ({e})")
        print(f"  Zapisuję jako: {alt_filepath}")
        with rasterio.open(alt_filepath, 'w', driver='GTiff', height=height, width=width,
                           crs=crs, transform=transform, count=count, dtype=dtype) as dst:
            dst.write(data, 1)
        return alt_filepath


def safe_geojson_write(gdf, filepath: str) -> str:
    """
    Bezpiecznie zapisuje plik GeoJSON. Jeśli plik jest zablokowany,
    próbuje zapisać pod alternatywną nazwą.

    Args:
        gdf: GeoDataFrame do zapisania
        filepath: Ścieżka do pliku wyjściowego

    Returns:
        Ścieżka do zapisanego pliku
    """
    try:
        gdf.to_file(filepath, driver='GeoJSON')
        print(f"  Zapisano: {filepath}")
        return filepath
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filepath)
        alt_path = f"{base}_{timestamp}{ext}"
        print(f"  Zapisuję jako: {alt_path}")
        gdf.to_file(alt_path, driver='GeoJSON')
        return alt_path


def load_raster_bands(path: str) -> tuple:
    """
    Wczytuje pasma rastrowe z pliku GeoTIFF.

    Args:
        path: Ścieżka do pliku rastrowego

    Returns:
        Tuple zawierająca:
        - all_bands: Numpy array ze wszystkimi pasmami (8, H, W)
        - ndvi: Obliczony indeks NDVI
        - transform: Transformacja geoprzestrzenna
        - crs: Układ współrzędnych
    """
    print("Reading image bands...")
    with rasterio.open(path) as src:
        cb = src.read(1).astype(float)   # pasmo coastal_blue
        b = src.read(2).astype(float)    # pasmo blue
        gi = src.read(3).astype(float)   # pasmo green_i
        g = src.read(4).astype(float)    # pasmo green
        y = src.read(5).astype(float)    # pasmo yellow
        r = src.read(6).astype(float)    # pasmo red
        re = src.read(7).astype(float)   # pasmo red_edge
        nir = src.read(8).astype(float)  # pasmo near_infra_red
        transform = src.transform
        crs = src.crs

    # Wszystkie pasma w jednej tablicy dla łatwiejszego dostępu
    all_bands = np.stack([cb, b, gi, g, y, r, re, nir], axis=0)

    # Oblicz NDVI
    ndvi = (nir - r) / (nir + r + 1e-6)

    return all_bands, ndvi, transform, crs

