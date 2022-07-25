import imageio.v2 as iio
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)


def main():
    n_clusters=3
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
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(images_flat)
    
    print(kmeans.cluster_centers_)
    
    print(kmeans.labels_)
    
    print(kmeans.labels_.shape)
    
    flat_seg = np.array(kmeans.labels_)
    
    print(f'unique: {np.unique(flat_seg)}')
    final_image = flat_seg.reshape(images.shape[1], images.shape[2])*255/(n_clusters-1)
    print(final_image)
    print(final_image.shape)
    iio.imwrite('sample.tif', final_image.astype(np.float32))
    
    # making images for individual clusters
    for clustervalue in np.unique(flat_seg):
        print(f'for cluster: {clustervalue}')
        idx = np.where(flat_seg==clustervalue)[0].tolist()
        # print(f'idx: {idx}')
        tmp_flat = np.zeros(flat_seg.shape)
        print(tmp_flat.shape)
        print(len(idx))
        print(type(idx))
        np.put(tmp_flat, idx, np.ones(len(idx)).tolist())
        print(tmp_flat.shape)
        tmp_img = np.reshape(tmp_flat, (images.shape[1], images.shape[2]))*255
        iio.imwrite(f'sample_cluster_{clustervalue}.png', tmp_img.astype(np.uint8))
    
    
    
    
    '''
    image_flat = image.flatten()
    print(f'Flatten_dim: {image_flat.shape}')
    image_flat = image_flat.reshape(-1, 1)
    print(image_flat)
    # return
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_flat)
    
    print(kmeans.cluster_centers_)
    
    print(kmeans.labels_)
    
    print(kmeans.labels_.shape)
    
    
    # plt.clf()
    # plt.axis("off")
    # plt.title(f"Quantized image ({n_clusters} colors, K-Means)")
    # plt.imshow(recreate_image(kmeans.cluster_centers_, kmeans.labels_, image.shape[0], image.shape[1]))
    # plt.show()
    
    flat_seg = np.array(kmeans.labels_)
    print(f'unique: {np.unique(flat_seg)}')
    final_image = flat_seg.reshape(image.shape[0], image.shape[1])*255/(n_clusters-1)
    print(final_image)
    print(final_image.shape)
    iio.imwrite('sample.tif', final_image.astype(np.float32))
    
    # making images for individual clusters
    for clustervalue in np.unique(flat_seg):
        print(f'for cluster: {clustervalue}')
        idx = np.where(flat_seg==clustervalue)[0].tolist()
        # print(f'idx: {idx}')
        tmp_flat = np.zeros(flat_seg.shape)
        print(tmp_flat.shape)
        print(len(idx))
        print(type(idx))
        np.put(tmp_flat, idx, np.ones(len(idx)).tolist())
        print(tmp_flat.shape)
        tmp_img = np.reshape(tmp_flat, (image.shape[0], image.shape[1]))*255
        iio.imwrite(f'sample_cluster_{clustervalue}.png', tmp_img.astype(np.uint8))
    '''
        
        
    
if __name__ == "__main__":
    main()