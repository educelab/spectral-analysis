from math import sqrt
import re
from typing import final
import imageio.v2 as iio
import numpy as np
import sklearn
import argparse
import sys

SUB_REGEX = r"(?P<h>\d+)x(?P<w>\d+)"
SUB_REGEX = re.compile(SUB_REGEX)


def parse_subdiv_params(sub_dim: str) -> tuple():
    '''
    Parse regexp for string HxW.
    Inputs:
            sub_dim: Dimension string for subdivision. Like 4x4, 3x3, 5x5, etc.
    Output:
            sub['h']: Number of subdivision along the height of the image.
            sub['w']: Number of subdivision along the width of the image.
            
    '''
    match = SUB_REGEX.match(sub_dim)
    if not match:
        print(f'Subdivision dimension cannot be matched.')
        return
    sub = match.groupdict()
    for key, value in sub.items():
        sub[key] = int(value)
    
    return sub['h'], sub['w']
    

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

def segment_subdivision(images: np.ndarray , n_clusters: int, sub_h: int = 4, sub_w: int = 4, n_batch: int = None) -> np.ndarray:
    '''
    Perform segmentation on subdivision.
    Inputs:
            images: Full images from the PCA bands.
            n_clusters: Number of clusters.
            sub_h: Number of subdivision along the height.
            sub_w: Number of subdivision along the width.
            n_batch: Batch size. Provide if number of features >=10000.
    Output:
            final_image: Finally clustered and stitched image.
    '''
    subimages = list()
    height = images.shape[1]
    width = images.shape[2]
    del_h = height//sub_h
    del_w = width//sub_w
    for h in range(sub_h):
        for w in range(sub_w):
            subimages.append(images[:,h*del_h:min((h+1)*del_h, height), w*del_w:min((w+1)*del_w, width)])
    # Apply kmeans on each subdivided image
    sub_seg = list()
    for subimg in subimages:
        sub_seg.append(segment_image(images=subimg, n_clusters=n_clusters, batch_size=n_batch))
    return stitch_images(image_list=sub_seg, sub_h=sub_h, sub_w=sub_w, height=height, width=width)
    
    
def stitch_images(image_list: list(), sub_h: int, sub_w: int, height: int, width: int) -> np.ndarray:
    '''
    Perform segmentation on subdivision.
    Inputs:
            image_list: List of locally segmented images.
            sub_h: Number of subdivision along the height.
            sub_w: Number of subdivision along the width.
            height: Height of the original un-subdivided image.
            width: Width of the original un-subdivided image.
    Output:
            stitched_image: Stitched image mosaic.
    '''
    del_h = height//sub_h
    del_w = width//sub_w
    stitched_image = np.zeros((height, width))
    print(f'shape of stitched output: {stitched_image.shape}')
    for i in range(sub_h):
        for j in range(sub_w):
            stitched_image[i*del_h:(i+1)*del_h, j*del_w:(j+1)*del_w] = image_list[(i*sub_w)+j]
    return stitched_image

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
                        help='Batch size Integer value. Default is sqrt(H*W*B).')
    parser.add_argument('--number-of-clusters', '-c', metavar='INT', type=int,
                        help='Number of clusters. Min. 2.', required=True)
    parser.add_argument('--subdivide', action='store_true',
                        help='If you want to divide the whole image into MxN grid and run segmentation on each of the grid and finally stitch it. Default is 4x4.')
    parser.add_argument('--subdivision-dimension', '-s', type=str,
                        help='Calculate the number of subdivision HxW. Default is 4x4.', default='4x4')
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
    
    # Initialising batch size.
    n_batch=args.batch_size
    
    # Initialise subdivision flag
    is_subdivide = args.subdivide
    
    # Initialising the file paths to be used for segmentation
    files = args.input_images
    
    # Making a list to store images from all the provided bands.
    images = list()
    for imagefile in files:
        images.append(iio.imread(imagefile))
    
    # Converting the list of images to numpy array (1D Array)
    images = np.array(images)
    
    # Perform segmentation
    if is_subdivide:
        (sub_h, sub_w) = parse_subdiv_params(args.subdivision_dimension)
        print(f'sub_h {sub_h}, sub_w: {sub_w}')
        print(f'Using Subdivision')
        final_image = segment_subdivision(images=images, n_clusters=n_clusters, n_batch=n_batch, sub_h=sub_h, sub_w=sub_w)
    else:
        print(f'Not using Subdivision')
        final_image = segment_image(images=images, n_clusters=n_clusters)
    
    print(f'Shape of the final_image: {final_image.shape}')
    
    #Saving the image. By default it is float 32 format
    iio.imwrite(f'{args.output_image}', final_image)
    
    
if __name__ == "__main__":
    main()