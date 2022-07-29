from math import sqrt
import re
import imageio.v2 as iio
import numpy as np
import sklearn
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

SUB_REGEX = r"(?P<h>\d+)x(?P<w>\d+)"
SUB_REGEX = re.compile(SUB_REGEX)

ROI_REGEX = r"(?P<x1>\d+)x(?P<y1>\d+)\+(?P<x2>\d+)x(?P<y2>\d+)"
ROI_REGEX = re.compile(ROI_REGEX)

test_cluster_list = ['kmeans', 'bkmeans']

def plot_inertia_vs_cluster(cluster_obj: list, K: range, algo: str, plot_name: str = 'plot-inertia_clusters') -> plt:
    plt.plot(K, cluster_obj,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title(f'Elbow Method For Optimal k for {algo}')
    plt.savefig(f'{plot_name}.png') 
    return plt

def parse_roi_params(roi_dim: str) -> tuple():
    '''
    Parse ROI regexp for string X1xY1+X2xY2.
    Inputs:
            roi_dim: Dimension string for ROI. Like 100x200+1000x2000, etc.
    Output:
            roi['x1']: Left top corner X-Coordinate.
            roi['y1']: Left top corner Y-Coordinate.
            roi['x2']: Right bottom corner X-Coordinate.
            roi['y2']: Right bottom corner Y-Coordinate.
            
    '''
    print(f'Recieved: {roi_dim}')
    match = ROI_REGEX.match(roi_dim)
    if not match:
        print(f'ROI dimension cannot be matched.')
        return
    roi = match.groupdict()
    print(f'ROI: {roi}')
    for key, value in roi.items():
        roi[key] = int(value)
    print(f'ROI: {roi}')
    return roi['x1'], roi['y1'], roi['x2'], roi['y2']

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

def spectral_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0) -> sklearn :
    '''
    Fit image to Spectral.
    Inputs:
            images_flat: Flattened PCA bands. Provide all the bands.
            n_clusters: Number of clusters.
            random_state: Seeding purpose.
            batch_size: Batch size. Provide if number of features >=10000.
    Output:
            spectral: Spectral clustering object
    '''
    from sklearn.cluster import SpectralCoclustering
    
    print(f'Using SpectralCoclustering')
    
    spectral = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    
    spectral.fit(images_flat)
    return spectral

def birch_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0) -> sklearn :
    '''
    Fit image to Spectral.
    Inputs:
            images_flat: Flattened PCA bands. Provide all the bands.
            n_clusters: Number of clusters.
            random_state: Seeding purpose.
            batch_size: Batch size. Provide if number of features >=10000.
    Output:
            spectral: Spectral clustering object
    '''
    from sklearn.cluster import Birch
    
    print(f'Using Birch')
    
    birch = Birch(n_clusters=n_clusters)
    
    birch.fit(images_flat)
    return birch

def kmeans_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0, batch_size: int = None) -> sklearn :
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
    
    # The documentation states that if the number of samples (in our case, the pixels), 
    # exceeds 10k, then use MiniBatchKMeans. This requires an extra input called batch size.
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

def bisecting_kmeans_fit(images_flat: np.ndarray, n_clusters: int, random_state: int = 0) -> sklearn :
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
    
    from sklearn.cluster import BisectingKMeans
    bkmeans = BisectingKMeans(n_clusters=n_clusters, random_state=random_state)
    
    bkmeans.fit(images_flat)
    return bkmeans

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

def subdivide_image(images: np.ndarray, sub_h: int = 4, sub_w: int = 4) -> list:
    subimages = list()
    height = images.shape[1]
    width = images.shape[2]
    del_h = height//sub_h
    del_w = width//sub_w
    for h in range(sub_h):
        for w in range(sub_w):
            subimages.append(images[:,h*del_h:min((h+1)*del_h, height), w*del_w:min((w+1)*del_w, width)])
    return subimages
    
    
def segment_subdivision(images: np.ndarray , n_clusters: int, algo: str = 'kmeans', sub_h: int = 4, sub_w: int = 4, n_batch: int = None) -> np.ndarray :
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
    
    #Getting the list of subdivided images
    subimages = subdivide_image(images=images, sub_h=sub_h, sub_w=sub_w)
    
    # Apply kmeans on each subdivided image
    sub_seg = list()
    for subimg in subimages:
        sub_seg.append(segment_image(images=subimg, n_clusters=n_clusters, batch_size=n_batch, algo=algo))
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

def cluster(images_flat: np.ndarray, n_clusters: int, n_batch: int = None, algo: str = 'kmeans') -> sklearn:
    # Perform clustering
    if algo == 'kmeans':
        seg_obj = kmeans_fit(images_flat, n_clusters=n_clusters, batch_size=n_batch)
    elif algo == 'spectral':
        seg_obj = spectral_fit(images_flat, n_clusters=n_clusters)
    elif algo == 'bkmeans':
        seg_obj = bisecting_kmeans_fit(images_flat, n_clusters=n_clusters)
    elif algo == 'birch':
        seg_obj = birch_fit(images_flat, n_clusters=None)
        
    return seg_obj
    
def segment_image(images: np.ndarray, n_clusters: int, algo: str = 'kmeans', batch_size: int = None) -> np.ndarray:
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
    
    # Clustering with required algorithm.
    seg_obj = cluster(images_flat=images_flat, n_clusters=n_clusters, algo=algo)
    
    
    # make a flattened numpy array
    flat_seg = np.array(seg_obj.labels_)
    
    #Number of clusters and its labels
    print(f'unique cluster labels: {np.unique(flat_seg)}')
    
    final_image = flat_seg.reshape(images.shape[1], images.shape[2])
    print(final_image)
    print(final_image.shape)
    
    return final_image
def get_matrix_idx(value: int, n_col: int) -> tuple :
    return ((value-(value%n_col))//n_col , value%n_col)

def eval_elbo(images: list, test_iter: int, algorithm: str, subdiv: bool = False, sub_dim: str = None, plot_name: str = 'elbo') -> None:
    
    K = range(3, test_iter + 1)
    if subdiv == False:
        clusters_inertia = list()
        for n_clusters in K:
            print(f'K: {n_clusters}')
            # Flatten image
            images_flat = flatten_image(images=images)
            # Clustering with required algorithm.
            seg_obj = cluster(images_flat=images_flat, n_clusters=n_clusters, algo=algorithm)
            clusters_inertia.append(seg_obj.inertia_)
        plot_inertia_vs_cluster(cluster_obj=clusters_inertia, K=K, algo=algorithm, plot_name=plot_name)
    else:
        if sub_dim is not None:
            print('subdivision present.')
            (sub_h, sub_w) = parse_subdiv_params(sub_dim=sub_dim)
            print(f'sub_h {sub_h}, sub_w: {sub_w}')
            #Making the plot matrix
            fig, axs = plt.subplots(nrows=sub_h, ncols=sub_w)
            sub_images = subdivide_image(images=images, sub_h=sub_h, sub_w=sub_w)
            sub_clusters = {}
            for idx, image in enumerate(sub_images):
                clusters_inertia = list()
                for n_clusters in K:
                    print(f'K: {n_clusters}')
                    # Flatten image
                    image_flat = flatten_image(images=image)
                    # Clustering with required algorithm.
                    seg_obj = cluster(images_flat=image_flat, n_clusters=n_clusters, algo=algorithm)
                    clusters_inertia.append(seg_obj.inertia_)
                sub_clusters[idx] = clusters_inertia
            for cluster_plot_idx in sub_clusters:
                splot=plot_inertia_vs_cluster(cluster_obj=clusters_inertia, K=K, algo=algorithm, plot_name=f'{plot_name}_{cluster_plot_idx}')
                (xr, xc) = get_matrix_idx(value=cluster_plot_idx, n_col=sub_w)
                axs[xr, xc].plot(K, clusters_inertia, 'bx-')
            fig.savefig(f'{plot_name}_all.png')
        else:
            print('sub-div dim required.')
            return

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
    parser.add_argument('--roi', type=str,
                          help='ROI used to calculate PCA: X1xY1+X2xY2')
    parser.add_argument('--algorithm', '-a', type=str, default='kmeans',
                          help='Choose Clustering Algorithm. Default is kmeans.', choices=['kmeans', 'spectral', 'bkmeans', 'birch'])
    parser.add_argument('--evaluate', '-e', type=int, default=None,
                          help='For testing purpose. Builds the graph for intertia vs clusters. Default is None. If it is provided with number, no output image will be there for the clustering. Only graph will be there.')
    parser.add_argument('--eval-subdiv', action='store_true',
                          help='For testing purpose. Evaluate over subdivided regions.')
    
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
    print(f'args-roi: {args.roi}')
    if args.roi is not None:
        (x1, y1, x2, y2) = parse_roi_params(args.roi)
        for imagefile in files:
            images.append(iio.imread(imagefile)[y1:y2, x1:x2])
    else:
        for imagefile in files:
            images.append(iio.imread(imagefile))
    
    # Converting the list of images to numpy array (1D Array)
    images = np.array(images)
    
    if args.evaluate is None:
        # Perform segmentation
        if is_subdivide:
            (sub_h, sub_w) = parse_subdiv_params(args.subdivision_dimension)
            print(f'sub_h {sub_h}, sub_w: {sub_w}')
            print(f'Using Subdivision')
            final_image = segment_subdivision(images=images, n_clusters=n_clusters, n_batch=n_batch, sub_h=sub_h, sub_w=sub_w, algo=args.algorithm)
        else:
            print(f'Not using Subdivision')
            final_image = segment_image(images=images, n_clusters=n_clusters, algo=args.algorithm)
        
        print(f'Shape of the final_image: {final_image.shape}')
        #Saving the image. By default it is float 32 format
        iio.imwrite(f'{args.output_image}', final_image)
    elif (args.evaluate is not None and args.algorithm in test_cluster_list):
        eval_elbo(images=images, test_iter=args.evaluate, algorithm=args.algorithm, subdiv=args.eval_subdiv, sub_dim=args.subdivision_dimension)
        
    
    
if __name__ == "__main__":
    main()