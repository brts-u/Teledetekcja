from geopandas import GeoDataFrame
from skimage.morphology import closing, opening, square
from shapely.geometry import LineString

from clusters import *

path = r"C:\Users\burb2\Desktop\Pliki Studia\Teledetekcja\grupa_6.tif"

print("Reading image bands...")
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
    print("Creating boolean mask...")
    from json_scraper import arr
    # Utwórz raster True/False na podstawie zakresów w arr
    result_mask = np.ones(cb.shape, dtype=bool)  # Inicjalizacja maski wynikowej

    for band, (min_val, mean_val, max_val) in zip([cb, b, gi, g, y, r, re, nir], arr):
        min_val *= 0.9             # Dostrojenie wartości
        max_val *= 1.1              # ale trochę na czuja
        result_mask &= (band >= min_val) & (band <= max_val)

    # Calculate NDVI and apply additional condition
    print("Applying NDVI mask...")
    ndvi = (nir - r) / (nir + r + 1e-6)
    vegetation_mask = np.logical_and(0.21 < ndvi, ndvi < 0.61) # trochę jestem zaskoczony jak wysokie są te wartości, ale takie działają
    result_mask &= vegetation_mask

    # Zapisz wynikowy raster True/False
    with rasterio.open('result_mask.tif', 'w', driver='GTiff', height=result_mask.shape[0],
                       width=result_mask.shape[1], count=1, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(result_mask.astype('uint8'), 1)
else:
    print("Using created boolean mask...")

print("Performing morphological operations...")
with rasterio.open('result_mask.tif') as src:
    img = src.read(1).astype(bool)  # Read as boolean
assert img is not None, "file could not be read, check with os.path.exists()"
kernel3 = square(3)
closed_img = closing(img, kernel3)

kernel2 = square(2)  # Equivalent to a 2x2 kernel
opened_img = opening(closed_img, kernel2)

labeled_img, num_features = label_with_diagonals(img)

print("Creating clusters...")
t0 = time.time()
clusters = create_clusters(labeled_img)
t1 = time.time()
print(f"Total clusters found: {len(clusters)}, {t1 - t0:.2f} seconds")

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
        sys.stdout.write(
            f"\rProgress: {percent}% (est. time remaining {elapsed * (100 - percent) / (percent + 1):.2f} seconds)")
        sys.stdout.flush()
        last_percent = percent

sys.stdout.write("\nDone!\n")  # Move to the next line after the loop

# save to tiff
with rasterio.open('depths.tif', 'w', driver='GTiff', height=labeled_img.shape[0], width=labeled_img.shape[1],
                   crs=crs, transform=transform, count=1, dtype='float32') as dst:
    depth_raster = np.zeros(labeled_img.shape, dtype='float32')
    for cluster in clusters.values():
        depth = cluster.depth
        for (row, col) in cluster.cells:
            depth_raster[row, col] = depth
    dst.write(depth_raster, 1)

clusters = list(clusters.values())
clusters.sort(key=lambda c: c.depth, reverse=True)

# Wektoryzacja
print("Starting vectorization...")
data = {'geometry': [], 'depth': []}
for cluster in clusters:
    # Jeśli depth < 130 to prawdopodobnie nie jest to obiekt, który nas obchodzi
    if cluster.depth < 130:
        continue

    full_path = cluster.longest_path
    steps = full_path[::10]
    geo_coords = []
    for row, col in steps:
        # transform * (col, row) gives (x, y) in georeferenced space
        # Note: col first, then row (x, y order)
        x, y = transform * (col, row)
        geo_coords.append((x, y))

    # Alternative: to get the center of the pixel instead of top-left corner
    geo_coords_center = []
    for row, col in steps:
        x, y = transform * (col + 0.5, row + 0.5)
        geo_coords_center.append((x, y))

    # Create a LineString from the georeferenced coordinates
    line = LineString(geo_coords)
    data['geometry'].append(line)
    data['depth'].append(cluster.depth)
gdf = GeoDataFrame(data, crs=crs)
gdf.to_file('train_tracks.geojson', driver='GeoJSON')





