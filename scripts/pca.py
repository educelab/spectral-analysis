import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA


def main():
    parser = argparse.ArgumentParser(description='Run PCA on a set of images')
    parser.add_argument('--input-files', '-i', nargs='+', required=True,
                        help='Input image files. All images must have the same '
                             'dimensions. Supports 8, 16, and 32-bit grayscale'
                             'images.')
    parser.add_argument('--output-dir', '-o', default='pca/',
                        help='Output directory')
    parser.add_argument('--components', '-c', type=int,
                        help='Number of components to compute. Must be in the '
                             'range [1, N] where N is the number of input '
                             'files. By default, will compute all components.')
    parser.add_argument('--incremental', action='store_true',
                        help='Run incremental PCA. Closely approximates PCA '
                             'through mini-batches. More memory efficient for '
                             'large datasets.')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for incremental PCA')
    args = parser.parse_args()

    # Check that there's at least 2 image paths
    if len(args.input_files) < 2:
        print(f'Error: Need 2+ input paths. Provided {len(args.input_files)}.')
        sys.exit(1)

    # Validate number of components
    if args.components is not None:
        if args.components < 1 or args.components > len(args.input_files):
            print(
                f'Error: Requested number of components invalid: {args.components}')
            sys.exit(1)
        components = args.components
    else:
        components = len(args.input_files)

    # Setup PCA
    if args.incremental:
        pca = IncrementalPCA(n_components=components,
                             batch_size=args.batch_size)
    else:
        pca = PCA(n_components=components)

    # Load images
    print('Loading images...')
    images = list()
    for i in args.input_files:
        images.append(iio.imread(i))

    # Convert to numpy array and flatten
    images = np.array(images)
    images_flat = images.reshape((images.shape[0], -1))
    images_flat = np.swapaxes(images_flat, 0, 1)

    # Fit and transform
    print('Fitting and transforming images...')
    transformed_flat = pca.fit_transform(images_flat)

    # Convert back to images
    transformed_flat = np.swapaxes(transformed_flat, 0, 1)
    pca_shape = (components,) + images.shape[1:]
    transformed_images = transformed_flat.reshape(pca_shape)

    # Save all images
    print('Saving images...')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    padding = len(str(components))
    for idx, img in enumerate(transformed_images):
        output_path = output_dir / f'pca_{idx:0{padding}}.tif'
        iio.imwrite(output_path, img.astype(np.float32))

    print('Done.')


if __name__ == '__main__':
    main()
