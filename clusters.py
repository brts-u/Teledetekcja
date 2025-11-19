import numpy as np
from scipy.ndimage import label
import warnings

def bbox_range(cluster):
    rows = np.any(cluster, axis=1)
    cols = np.any(cluster, axis=0)
    if not rows.any() or not cols.any():
        return None  # No cluster found
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_max - x_min, y_max - y_min

def keep_long_clusters(binary_image):
    warnings.warn("To trwa bardzo długo! (nie wiem nawet ile bo tego nie odpalałem na całym rastrze)", UserWarning)
    labeled_raster, num_features = label(binary_image)

    for cluster_id in range(1, num_features + 1):
        x_range, y_range = bbox_range(labeled_raster == cluster_id)
        if x_range < 50 or y_range < 50:
            binary_image[labeled_raster == cluster_id] = 0

    return binary_image, labeled_raster

if __name__ == "__main__":
    import rasterio

    with rasterio.open('maly_raster.tif') as src:
        img = src.read(1).astype(bool)  # Read as boolean
        transform = src.transform
        crs = src.crs

    assert img is not None, "file could not be read, check with os.path.exists()"

    filtered_img, labeled = keep_long_clusters(img)

    with rasterio.open('labeled_clusters.tif', 'w', driver='GTiff', height=labeled.shape[0],
                       width=labeled.shape[1], count=1, dtype='uint16', crs=crs, transform=transform) as dst:
        dst.write(labeled.astype('uint16'), 1)

    with rasterio.open('clusters.tif', 'w', driver='GTiff', height=filtered_img.shape[0],
                       width=filtered_img.shape[1], count=1, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(filtered_img.astype('uint8'), 1)