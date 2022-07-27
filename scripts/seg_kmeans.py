from math import sqrt
import imageio.v2 as iio
import numpy as np
import sklearn
import argparse
import sys

def kmeans_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0, batch_size: int = None) -> sklearn:
    '''
    Fit image to kmeans.
    Inputs:
            images_flat: Flattened PCA bands. Provide all the bands.
            n_clusters: Number of clusters.
            random_state: Seeding purpose.
            batch_size: Batch size. Provide if number of features >=10000.
    Output:
            kmeans: Kmeans object
    '''
    n_features = images_flat.shape[0]
    if n_features <= 10000:
        print(f'Using KMeans')
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        from sklearn.cluster import MiniBatchKMeans
        print(f'Using MiniBatchKMeans')
        if batch_size == None:
            n_batch_size = round(sqrt(images_flat.shape[0]*images_flat.shape[1]))
            print(f'Using default batch_size of {n_batch_size}')
        else:
            n_batch_size = batch_size
            print(f'Using custom batch_size of {n_batch_size}')
            
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=n_batch_size)
    
    kmeans.fit(images_flat)
    return kmeans

def flatten_image(images: np.ndarray) -> np.array :
    '''
    Flatten image.
    Inputs:
            images: Images of PCA bands.
    Output:
            images_flat: Flattened image numpy array.
    '''
    images_flat = images.reshape(images.shape[0], -1) # (c, w, h) -> (c, w*h)
    images_flat = np.swapaxes(images_flat, 0, 1) # (c, w*h) -> (w*h, c)
    
    return images_flat

def segment_image(images: np.ndarray, n_clusters: int, batch_size: int = None) -> np.ndarray:
    '''
    Segment image using KMeans.
    Inputs:
            images: Images of PCA bands.
            n_clusters: Number of clusters.
            batch_size: Batch size. Provide if number of features >=10000.
    Output:
            final_image: Segmented image in numpy array.
    '''
    # Flatten image
    images_flat = flatten_image(images=images)
    
    # Perform kmeans
    kmeans = kmeans_fit(images_flat, n_clusters=n_clusters, batch_size=batch_size)
    
    # make a flattened numpy array
    flat_seg = np.array(kmeans.labels_)
    
    #Number of clusters and its labels
    print(f'unique cluster labels: {np.unique(flat_seg)}')
    
    final_image = flat_seg.reshape(images.shape[1], images.shape[2])
    print(final_image)
    print(final_image.shape)
    
    return final_image

def main():
    parser = argparse.ArgumentParser(description='Run KMeans for segmentation on a set of images')
    parser.add_argument('--input-images', '-i', nargs='+', metavar='IMAGE',
                        help='List of input image files. All images must have '
                             'the same dimensions. Supports 8, 16, and 32-bit '
                             'grayscale images', required=True)
    parser.add_argument('--output-image', '-o', default='sample.tif', metavar='FILE',
                        help='Output segmented image name')
    parser.add_argument('--batch-size', '-b', default=None, metavar='INT', type=int,
                        help='Batch size Integer value. Default is sqrt(W*H*B).')
    parser.add_argument('--number-of-clusters', '-c', metavar='INT', type=int,
                        help='Number of clusters. Min. 2.', required=True)
    args = parser.parse_args()
    
    # Validation for number of k-means clusters.
    if args.number_of_clusters <= 1:
        print(f'Cluster size found {args.number_of_clusters}. Required >=2')
        sys.exit(1)
    
    # Validation for number of batches provided.
    if args.batch_size != None and args.batch_size <= 1:
        print(f'Batch size found {args.batch_size}. Required >=1')
        sys.exit(1)
    
    # Initialising number fo kmeans cluster provided by the user.
    n_clusters=args.number_of_clusters
    
    # Initialising the file paths to be used for segmentation
    files = args.input_images
    
    # Making a list to store images from all the provided bands.
    images = list()
    for imagefile in files:
        images.append(iio.imread(imagefile))
    
    # Converting the list of images to numpy array (1D Array)
    images = np.array(images)
    
    # Perform segmentation
    final_image = segment_image(images=images, n_clusters=n_clusters)
    
    #Saving the image. By default it is float 32 format
    iio.imwrite(f'{args.output_image}', final_image)
    
    
if __name__ == "__main__":
    main()