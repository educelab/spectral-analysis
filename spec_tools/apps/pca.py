import argparse
import logging
import pickle
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from educelab import imgproc

import spec_tools.pca as pca
from spec_tools.utils.apps import parse_roi_params, setup_logging


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
    parser.add_argument('--output-format', '-f', metavar='FORMAT',
                        default='tif', choices=['png', 'jpg', 'tif'],
                        type=str.lower)

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
    pca_opts.add_argument('--roi', type=str,
                          help='ROI used to calculate PCA: WxH+X+Y')
    pca_opts.add_argument('--save-model', metavar='FILE.pickle',
                          help='Save a pickled PCA instance to a file')
    pca_opts.add_argument('--load-model', metavar='FILE.pickle',
                          help='Load a pickled PCA instance from a file')
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger('spec-pca')

    # Get bools
    load_inputs = args.input_images is not None
    load_transforms = args.images_to_transform is not None
    load_model = args.load_model is not None

    # Check for the required parameters
    if not load_inputs and not load_model:
        logger.error(f'Requires either --input-images/--load-model')
        sys.exit(1)

    if load_model and not load_inputs and not load_transforms:
        logger.error(f'Loading PCA model but did not specify files to be '
                     f'transformed. Must provide either '
                     f'--input-images/--images-to-transform')
        sys.exit(1)

    # Load input images if we're loading PCA but don't have transform files
    if not load_model or not load_transforms:
        logger.info('Loading input images')
        images = list()
        for i in args.input_images:
            images.append(iio.imread(i))
        images = np.array(images)

    # Load or compute PCA
    if load_model:
        logger.info('Loading pickled PCA model')
        with Path(args.load_model).open('rb') as file:
            pca_model = pickle.load(file)
    else:
        # Get ROI parameters
        roi = None
        if args.roi is not None:
            roi = parse_roi_params(args.roi)
        logger.info('Fitting the model')
        pca_model = pca.fit(images,
                            components=args.components,
                            batch_size=args.batch_size,
                            incremental=args.incremental,
                            roi=roi)

    # Save the PCA file
    if args.save_model is not None:
        logger.info('Saving pickled PCA model')
        with Path(args.save_model).open('wb') as file:
            pickle.dump(pca_model, file)

    # Validate number of transform files matches number of components
    components = pca_model.components_.shape[0]
    if load_transforms and len(args.images_to_transform) != components:
        logger.error(f'Number of files to be transformed '
                     f'({len(args.images_to_transform)}) doesn\'t match the '
                     f'number of components ({components})')
        sys.exit(1)

    # Load the transform images
    if args.images_to_transform is None:
        transform_images = images
    else:
        logger.info('Loading images to transform')
        transform_images = list()
        for i in args.images_to_transform:
            transform_images.append(iio.imread(i))
        transform_images = np.array(transform_images)

    # Transform images
    logging.info('Transforming images')
    transformed_images = pca.apply_transform(transform_images, pca_model)

    # Save all images
    logger.info('Saving images')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    padding = len(str(components))
    fmt = args.output_format
    for idx, img in enumerate(transformed_images):
        if fmt in ('png', 'jpg'):
            img = imgproc.normalize(img)
            img = imgproc.as_dtype(img, np.uint8)
        else:
            img = img.astype(np.float32)
        output_path = output_dir / f'{args.prefix}{idx:0{padding}}.{fmt}'
        iio.imwrite(output_path, img)

    logger.info('Done')


if __name__ == '__main__':
    main()
