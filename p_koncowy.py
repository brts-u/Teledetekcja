import rasterio
import numpy as np
from skimage.filters import gabor
from skimage.feature import canny
from skimage.morphology import skeletonize, closing, opening, square
from skimage.transform import probabilistic_hough_line
import geopandas as gpd
from shapely.geometry import LineString

# ----------------------------
# 1. Wczytanie obrazu (8 pasm)
# ----------------------------
path = r"C:\Users\burb2\Desktop\Pliki Studia\Teledetekcja\grupa_6.tif"

with rasterio.open(path) as src:
    cb = src.read(1).astype(float)  # pasmo coastal_blue
    b = src.read(2).astype(float)  # pasmo blue
    gi = src.read(3).astype(float)  # pasmo green_i
    g = src.read(4).astype(float)  # pasmo green
    y = src.read(5).astype(float)  # pasmo yellow
    r = src.read(6).astype(float)  # pasmo red
    re = src.read(7).astype(float)  # pasmo red_edge
    nir = src.read(8).astype(float)  # pasmo near_infra_red
    transform = src.transform
    crs = src.crs

# Pomijanie tworzenia maski, jeśli jest już utworzona
SKIP_BOOLEAN_MASK = True

if not SKIP_BOOLEAN_MASK:
    from json_scraper import arr
    # Utwórz raster True/False na podstawie zakresów w arr
    result_mask = np.ones(cb.shape, dtype=bool)  # Inicjalizacja maski wynikowej

    for band, (min_val, mean_val, max_val) in zip([cb, b, gi, g, y, r, re, nir], arr):
        max_val *= 1.1
        result_mask &= (band >= min_val) & (band <= max_val)

    # Zapisz wynikowy raster True/False
    with rasterio.open('result_mask.tif', 'w', driver='GTiff', height=result_mask.shape[0],
                       width=result_mask.shape[1], count=1, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(result_mask.astype('uint8'), 1)

with rasterio.open('result_mask.tif') as src:
    img = src.read(1).astype(bool)  # Read as boolean
assert img is not None, "file could not be read, check with os.path.exists()"
kernel3 = square(3)
closed_img = closing(img, kernel3)

kernel2 = square(2)  # Equivalent to a 2x2 kernel
opened_img = opening(closed_img, kernel2)

with rasterio.open('complete_mask.tif', 'w', driver='GTiff', height=opened_img.shape[0],
                   width=opened_img.shape[1], count=1, dtype='uint8', crs=crs, transform=transform) as dst:
    dst.write(opened_img.astype('uint8'), 1)

# ----------------------------
# DALEJ CZAT
# ----------------------------

# # Normalizacja
# r /= r.max()
# nir /= nir.max()
# re /= re.max()
#
# # ----------------------------
# # 2. NDVI + maska niskiej roślinności
# # ----------------------------
# ndvi = (nir - r) / (nir + r + 1e-6)
# low_veg_mask = ndvi < 0.25  # <--- można dostroić (0.20 – 0.35)
#
# # ----------------------------
# # 3. Wzmocnienie cech liniowych (Gabor)
# # ----------------------------
# gabor_response, _ = gabor(r, frequency=0.2)  # start
# gabor_norm = (gabor_response - gabor_response.min()) / (gabor_response.max() - gabor_response.min())
#
# candidate = gabor_norm * low_veg_mask * opened_img

#
# # ----------------------------
# # 4. Canny + morfologia
# # ----------------------------
# edges = canny(candidate, sigma=1.5)
# edges_closed = closing(edges, square(3))
# skel = skeletonize(edges_closed)
#
# # ----------------------------
# # 5. Hough transform → linie
# # ----------------------------
# lines = probabilistic_hough_line(
#     skel,
#     threshold=10,
#     line_length=40,  # <--- zmieniać w zależności od regionu
#     line_gap=5
# )
#
# # ----------------------------
# # 6. Wektoryzacja
# # ----------------------------
# geoms = []
# for p0, p1 in lines:
#     (x0, y0) = rasterio.transform.xy(transform, p0[1], p0[0])
#     (x1, y1) = rasterio.transform.xy(transform, p1[1], p1[0])
#     geoms.append(LineString([(x0, y0), (x1, y1)]))
#
# gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
# gdf.to_file("torowiska_wykryte.gpkg", driver="GPKG")
#
# print("Zakończono! Wynik zapisano: torowiska_wykryte.shp")
