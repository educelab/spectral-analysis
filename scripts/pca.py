import argparse
import pickle
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA


def pca_fit(x, components: int = None, batch_size: int = None,
            incremental: bool = False):
    # Validate number of components
    if components is not None and not (1 < components < x.shape[0]):
        print(
            f'Error: Requested components ({components}) outside range [1, {x.shape[0]}]')
        sys.exit(1)

    # Setup new PCA
    if incremental:
        pca = IncrementalPCA(n_components=components,
                             batch_size=batch_size)
    else:
        pca = PCA(n_components=components)

    # Flatten input
    x_flat = x.reshape((x.shape[0], -1))
    x_flat = np.swapaxes(x_flat, 0, 1)

    # Fit input files
    print(f'Fitting {x.shape[0]} images...')
    pca.fit(x_flat)
    return pca


def main():
    parser = argparse.ArgumentParser(description='Run PCA on a set of images')
    parser.add_argument('--input-images', '-i', nargs='+', metavar='IMAGE',
                        help='List of input image files. All images must have '
                             'the same dimensions. Supports 8, 16, and 32-bit '
                             'grayscale images')
    parser.add_argument('--images-to-transform', nargs='+', metavar='IMAGE',
                        help='Set of images to which dimensionality reduction '
                             'will be applied. If not specified, will default '
                             'to the list provided by --input-images')
    parser.add_argument('--output-dir', '-o', default='pca/', metavar='DIR',
                        help='Output directory for transformed images')
    parser.add_argument('--output-prefix', dest='prefix', default='pca_', )

    pca_opts = parser.add_argument_group('pca options')
    pca_opts.add_argument('--incremental', action='store_true',
                          help='Run incremental PCA. Closely approximates PCA '
                               'through mini-batches. More memory efficient '
                               'for large datasets.')
    pca_opts.add_argument('--components', '-c', type=int, metavar='INT',
                          help='Number of components to compute. Must be in '
                               'the range [1, N] where N is the number of '
                               'input files. By default, will compute all '
                               'components.')
    pca_opts.add_argument('--batch-size', type=int, metavar='INT',
                          help='Batch size for incremental PCA')
    pca_opts.add_argument('--save-pca', metavar='FILE.pickle',
                          help='Save a pickled PCA instance to a file')
    pca_opts.add_argument('--load-pca', metavar='FILE.pickle',
                          help='Load a pickled PCA instance from a file')
    args = parser.parse_args()

    # Get bools
    load_inputs = args.input_images is not None
    load_transforms = args.images_to_transform is not None
    load_pca = args.load_pca is not None

    # Check for the required parameters
    if not load_inputs and not load_pca:
        print(f'Error: Requires either --input-images/--load-pca')
        sys.exit(1)

    if load_pca and not load_inputs and not load_transforms:
        print(f'Error: Loading PCA but did not specify files to be '
              f'transformed. Must provide either '
              f'--input-images/--images-to-transform')
        sys.exit(1)

    # Load input images if we're loading PCA but don't have transform files
    if not load_pca or not load_transforms:
        print('Loading input images...')
        images = list()
        for i in args.input_images:
            images.append(iio.imread(i))
        images = np.array(images)

    # Load or compute PCA
    if load_pca:
        print('Loading pickled PCA...')
        with Path(args.load_pca).open('rb') as file:
            pca = pickle.load(file)
    else:
        pca = pca_fit(images,
                      components=args.components,
                      batch_size=args.batch_size,
                      incremental=args.incremental)

    # Save the PCA file
    if args.save_pca is not None:
        print('Saving pickled PCA...')
        with Path(args.save_pca).open('wb') as file:
            pickle.dump(pca, file)

    # Validate number of transform files matches number of components
    components = pca.n_components_
    if load_transforms and len(args.images_to_transform) != components:
        print(f'Error: Number of files to be transformed '
              f'({len(args.images_to_transform)}) doesn\'t match the number of '
              f'components ({components})')
        sys.exit(1)

    # Load the transform images
    if args.images_to_transform is None:
        transform_images = images
    else:
        print('Loading images to transform...')
        transform_images = list()
        for i in args.images_to_transform:
            transform_images.append(iio.imread(i))
        transform_images = np.array(transform_images)

    # Transform images
    print(f'Transforming {transform_images.shape[0]} images...')
    transformed_flat = transform_images.reshape((transform_images.shape[0], -1))
    transformed_flat = np.swapaxes(transformed_flat, 0, 1)
    transformed_flat = pca.transform(transformed_flat)

    # Convert back to images
    transformed_flat = np.swapaxes(transformed_flat, 0, 1)
    pca_shape = (components,) + transform_images.shape[1:]
    transformed_images = transformed_flat.reshape(pca_shape)

    # Save all images
    print('Saving images...')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    padding = len(str(components))
    for idx, img in enumerate(transformed_images):
        output_path = output_dir / f'{args.prefix}{idx:0{padding}}.tif'
        iio.imwrite(output_path, img.astype(np.float32))

    print('Done.')


if __name__ == '__main__':
    main()
