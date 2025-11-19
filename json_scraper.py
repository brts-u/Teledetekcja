import json
import numpy as np

# Wczytaj dane z pliku
with open('stats_grupa_6.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Wypisz properties jedynego feature
properties = data['features'][0]['properties']

ids = ['1CB', '2B', '3Gi', '4G', '5Y', '6R', '7RE', '8NIR']
arr = np.zeros((8, 3))
for id, row in zip(ids, arr):
    row[0] = properties[f'{id}mean'] - properties[f'{id}stdev']
    row[1] = properties[f'{id}mean']
    row[2] = properties[f'{id}mean'] + properties[f'{id}stdev']

if __name__ == "__main__":
    import rasterio

    path = r"C:\Users\burb2\Desktop\Pliki Studia\Teledetekcja\grupa_6.tif"

    with rasterio.open(path) as src:
        cb = src.read(1).astype(float)     # pasmo 1
        b = src.read(2).astype(float)     # pasmo 2
        gi = src.read(3).astype(float)     # pasmo 3
        g = src.read(4).astype(float)     # pasmo 4
        y = src.read(5).astype(float)     # pasmo 5
        r = src.read(6).astype(float)     # pasmo 6
        re = src.read(7).astype(float)     # pasmo 7
        nir = src.read(8).astype(float)     # pasmo 8
        transform = src.transform
        crs = src.crs

    # Utwórz raster True/False na podstawie zakresów w arr
    result_mask = np.ones(cb.shape, dtype=bool)  # Inicjalizacja maski wynikowej

    for band, (min_val, mean_val, max_val) in zip([cb, b, gi, g, y, r, re, nir], arr):
        result_mask &= (band >= min_val) & (band <= max_val)

    # Zapisz wynikowy raster True/False
    with rasterio.open('result_mask.tif', 'w', driver='GTiff', height=result_mask.shape[0],
                       width=result_mask.shape[1], count=1, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(result_mask.astype('uint8'), 1)