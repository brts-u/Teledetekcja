import rasterio
import numpy as np
from skimage.filters import gabor
from skimage.feature import canny
from skimage.morphology import skeletonize, closing, square
from skimage.transform import probabilistic_hough_line
import geopandas as gpd
from shapely.geometry import LineString

# ----------------------------
# 1. Wczytanie obrazu (8 pasm)
# ----------------------------
path = r"C:\Users\burb2\Desktop\Pliki Studia\Teledetekcja\grupa_7.tif"

with rasterio.open(path) as src:
    red = src.read(6).astype(float)     # pasmo 6
    nir = src.read(8).astype(float)     # pasmo 8
    red_edge = src.read(7).astype(float) # opcjonalnie
    transform = src.transform
    crs = src.crs

# Normalizacja
red /= red.max()
nir /= nir.max()
red_edge /= red_edge.max()

# ----------------------------
# 2. NDVI + maska niskiej roślinności
# ----------------------------
ndvi = (nir - red) / (nir + red + 1e-6)
low_veg_mask = ndvi < 0.25  # <--- można dostroić (0.20–0.35)

# ----------------------------
# 3. Wzmocnienie cech liniowych (Gabor)
# ----------------------------
gabor_response, _ = gabor(red, frequency=0.2)  # start
gabor_norm = (gabor_response - gabor_response.min()) / (gabor_response.max() - gabor_response.min())

candidate = gabor_norm * low_veg_mask

# ----------------------------
# 4. Canny + morfologia
# ----------------------------
edges = canny(candidate, sigma=1.5)
edges_closed = closing(edges, square(3))
skel = skeletonize(edges_closed)

# ----------------------------
# 5. Hough transform → linie
# ----------------------------
lines = probabilistic_hough_line(
    skel,
    threshold=10,
    line_length=40,  # <--- zmieniać w zależności od regionu
    line_gap=5
)

# ----------------------------
# 6. Wektoryzacja
# ----------------------------
geoms = []
for p0, p1 in lines:
    (x0, y0) = rasterio.transform.xy(transform, p0[1], p0[0])
    (x1, y1) = rasterio.transform.xy(transform, p1[1], p1[0])
    geoms.append(LineString([(x0, y0), (x1, y1)]))

gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
gdf.to_file("torowiska_wykryte.gpkg", driver="GPKG")

print("Zakończono! Wynik zapisano: torowiska_wykryte.shp")
