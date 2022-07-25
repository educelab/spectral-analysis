from math import sqrt
import imageio.v2 as iio
import numpy as np
from pathlib import Path

def kmeans_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0):
    n_features = images_flat.shape[0]
    if n_features <= 10000:
        print(f'Using KMeans')
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        from sklearn.cluster import MiniBatchKMeans
        print(f'Using MiniBatchKMeans')
        n_batch_size = round(sqrt(images_flat.shape[0]*images_flat.shape[1]))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=n_batch_size)
    
    kmeans.fit(images_flat)
    return kmeans

def main():
    n_batch_size=1000
    n_clusters=5
    data = Path(f'data/HolyTrinity_01v_Test/pca_seg/')
    files = [x for x in data.iterdir()]
    images = []
    for imagefile in files:
        images.append(iio.imread(imagefile))
    images = np.array(images)
    
    images_flat = images.reshape(images.shape[0], -1) # (c, w, h) -> (c, w*h)
    images_flat = np.swapaxes(images_flat, 0, 1) # (c, w*h) -> (w*h, c)
    
    
    kmeans = kmeans_fit(images_flat, n_clusters=n_clusters)
    
    flat_seg = np.array(kmeans.labels_)
    
    print(f'unique cluster labels: {np.unique(flat_seg)}')
    
    final_image = flat_seg.reshape(images.shape[1], images.shape[2])
    print(final_image)
    print(final_image.shape)
    iio.imwrite('sample.tif', final_image)
    
    
if __name__ == "__main__":
    main()