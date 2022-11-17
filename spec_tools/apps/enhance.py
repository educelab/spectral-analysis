import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from skimage import (exposure, img_as_float, io)
from tqdm import tqdm

import spec_tools.utils.exiftool as exiftool
from spec_tools.utils.apps import (expand_path_list, setup_logging,
                                   to_numpy_dtype)
from spec_tools.utils.image import as_dtype

VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_SUFFIX = ''
ENHANCE_VERSION = f'{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}{VERSION_SUFFIX}'


def main():
    # Parse args
    parser = argparse.ArgumentParser('spec-enhance')
    parser.add_argument('--input-images', '-i', nargs='+', metavar='IMAGE',
                        required=True,
                        help='List of input image files. All images must have '
                             'the same dimensions. Supports 8, 16, and 32-bit '
                             'grayscale images')
    parser.add_argument('--output-format', '-f', metavar='EXT', type=str.lower,
                        help='Output file extension supported by imageio '
                             '(e.g. \'jpg\' or \'tif\')')
    parser.add_argument('--output-depth', '-d', metavar='INT',
                        type=to_numpy_dtype,
                        help='Specify the preferred output bits-per-channel. '
                             'By default, the output depth matches the input '
                             'depth. Supported depths: 8, 16, 32')
    parser.add_argument('--output-dir', '-o', default='processed/',
                        metavar='DIR',
                        help='Output directory for transformed images')
    parser.add_argument('--suffix-separator', type=str, metavar='STR',
                        help='Optional separator between the original input '
                             'filename and the enhancement\'s suffix. For '
                             'example, a separator of \'_\' would result in '
                             'outputs \'foo_N.jpg\' or \'bar_E.tif\'')
    parser.add_argument('--metadata', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Copy metadata tags from the input file to the '
                             'output file')
    parser.add_argument('--progress', '-P',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Display progress bars')

    enhance_opts = parser.add_argument_group('basic enhancement options')
    enhance_opts.add_argument('--enhance', '-e',
                              action=argparse.BooleanOptionalAction,
                              help='If enabled, enhance image contrast using '
                                   'gamma correction and contrast stretching')
    enhance_opts.add_argument('--gamma', metavar='FLOAT', type=float,
                              default=0.4545,
                              help='Gamma correction term. Gamma < 1.0 '
                                   'brightens the image while gamma > 1.0 '
                                   'darkens it')
    enhance_opts.add_argument('--stretch-min', metavar='FLOAT', type=float,
                              default=0, help='Stretch minimum percentile')
    enhance_opts.add_argument('--stretch-max', metavar='FLOAT', type=float,
                              default=99, help='Stretch maximum percentile')

    clahe_opts = parser.add_argument_group('CLAHE options')
    clahe_opts.add_argument('--clahe', '-c',
                            action=argparse.BooleanOptionalAction,
                            help='If enabled, enhance the image using Contrast '
                                 'Limited Adaptive Histogram Equalization '
                                 '(CLAHE) after all previous enhancements')
    clahe_opts.add_argument('--clahe-kernel', type=int, metavar='INT',
                            help='Size of the CLAHE kernel in each dimension. '
                                 'See the scikit-image documentation for CLAHE'
                                 'for more details')
    clahe_opts.add_argument('--clahe-bins', type=int, default=256,
                            metavar='INT',
                            help='Number of gray bins in the histogram')
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger('spec-enhance')

    # Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default suffix
    if args.suffix_separator is None:
        suffix_sep = ''
    else:
        suffix_sep = args.suffix_separator
    suffix = 'N'
    if args.enhance:
        suffix = 'E'
    if args.clahe:
        suffix = f'{suffix}C'

    # Iterate the files
    input_images = expand_path_list(args.input_images)
    if len(input_images) == 0:
        logger.warning('Nothing to process')
        sys.exit()

    for img_path in tqdm(input_images, desc=f'Enhancing images ({suffix})',
                         disable=not args.progress):
        # Check if file exists
        if not img_path.exists():
            logger.error(f'File does not exist: {img_path}')
            continue

        # Load image
        try:
            img = io.imread(img_path)
        except ValueError:
            logger.error(f'Failed to load file: {str(img_path)}')
            continue
        in_dtype = img.dtype

        # Convert to float for processing
        img = img_as_float(img)

        # Auto-level
        img = exposure.rescale_intensity(img)

        # Enhance
        if args.enhance:
            # Gamma
            img = exposure.adjust_gamma(img, args.gamma)

            # Contrast stretch
            per_min = np.percentile(img, args.stretch_min)
            per_max = np.percentile(img, args.stretch_max)
            img = exposure.rescale_intensity(img, (per_min, per_max))

        # CLAHE
        if args.clahe:
            # Run CLAHE
            img = exposure.equalize_adapthist(img,
                                              kernel_size=args.clahe_kernel,
                                              nbins=args.clahe_bins)

        # Determine output format
        out_fmt = args.output_format if args.output_format is not None else img_path.suffix
        out_fmt = f'.{out_fmt}' if out_fmt[0] != '.' else out_fmt
        kwargs = {}

        # Type conversion
        if out_fmt in ['.bmp', '.jpg', '.jpeg']:
            out_dtype = np.uint8
        elif args.output_depth is not None:
            out_dtype = args.output_depth
        else:
            out_dtype = in_dtype
        img = as_dtype(img, out_dtype)

        # Format specific opts
        if out_fmt in ['.jpg', '.jpeg']:
            kwargs['quality'] = 100
        elif out_fmt in ['.tif', '.tiff']:
            kwargs['compression'] = 'zlib'
            kwargs['compressionargs'] = {'level': 9}

        # Save the image to disk
        out_file = f'{img_path.stem}{suffix_sep}{suffix}{out_fmt}'
        out_path = out_dir / out_file
        io.imsave(out_path, img, **kwargs)

        # Copy metadata
        if args.metadata:
            extra_tags = [f'-Software=EduceLab spec-enhance v{ENHANCE_VERSION}']
            exiftool.copy_all(img_path, out_path, extra_tags=extra_tags)


if __name__ == '__main__':
    main()
