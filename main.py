import geopandas as gpd
from osgeo import gdal
import shapely
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\User\OneDrive - Politechnika Warszawska\3 rok\Teledetekcja\projekt_3\grupa_6.tif"

gdal.UseExceptions()

dataset: gdal.Dataset = gdal.Open(path)
affine = dataset.GetGeoTransform()

band_r = dataset.GetRasterBand(6).ReadAsArray()
band_g = dataset.GetRasterBand(4).ReadAsArray()
band_b = dataset.GetRasterBand(2).ReadAsArray()
band_ir = dataset.GetRasterBand(8).ReadAsArray()

rgb_array = np.dstack((band_r, band_g, band_b))
rgb_array = (rgb_array - np.min(rgb_array[rgb_array > 0])) / (np.max(rgb_array) - np.min(rgb_array[rgb_array > 0]))

plt.figure(figsize=(10, 8))
plt.imshow(rgb_array)
plt.title('RGB')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()