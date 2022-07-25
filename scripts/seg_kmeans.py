from math import sqrt
import imageio.v2 as iio
import numpy as np
from pathlib import Path



def main():
    n_batch_size=1000
    n_clusters=5
    # data = f'data/HolyTrinity_01v_Test/PCA/HolyTrinity_01v_Test_A_pca/HolyTrinity_01v_Test_A_pca_00.tif'
    data = Path(f'data/HolyTrinity_01v_Test/pca_seg/')
    files = [x for x in data.iterdir()]
    images = []
    for imagefile in files:
        images.append(iio.imread(imagefile))
    images = np.array(images)
    print(images.shape)
    images_flat = images.reshape(images.shape[0], -1)
    images_flat = np.swapaxes(images_flat, 0, 1)
    n_samples = images_flat.shape[0]
    
    if n_samples <= 10000:
        print(f'Using KMeans')
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    else:
        from sklearn.cluster import MiniBatchKMeans
        print(f'Using MiniBatchKMeans')
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=round(sqrt(n_samples)))
    
    kmeans.fit(images_flat)
    
    print(kmeans.cluster_centers_)
    
    print(kmeans.labels_)
    
    print(kmeans.labels_.shape)
    
    flat_seg = np.array(kmeans.labels_)
    
    print(f'unique: {np.unique(flat_seg)}')
    final_image = flat_seg.reshape(images.shape[1], images.shape[2])
    print(final_image)
    print(final_image.shape)
    iio.imwrite('sample.tif', final_image)
    
    
if __name__ == "__main__":
    main()