import argparse
import json
import logging
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import numpy as np
from educelab import imgproc
from educelab.imgproc import exiftool, pipeline
from skimage import img_as_float
from tqdm import tqdm
import imageio.v3 as iio

from spec_tools.utils.apps import (expand_path_list, setup_logging,
                                   to_numpy_dtype)

VERSION_MAJOR = 1
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION_SUFFIX = ''
ENHANCE_VERSION = f'{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}{VERSION_SUFFIX}'

def main():
    # Parse args
    parser = argparse.ArgumentParser('spec-enhance',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-images', '-i', nargs='+', metavar='IMAGE',
                        required=True,
                        help='List of input image files. Supports 8, 16, and '
                             '32-bit grayscale images')
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
    parser.add_argument('--suffix', type=str, metavar='STR', default='_E')
    parser.add_argument('--metadata', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Copy metadata tags from the input file to the '
                             'output file')
    parser.add_argument('--logfile', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Write a configuration log file (JSON) to the '
                             'output directory')
    parser.add_argument('--progress', '-P',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Display progress bars')
    parser.add_argument('--quality', '-q', type=int,
                        help='Output format-specific compression level')
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    enhance_opts = pipeline.add_parser_enhancement_group(parser)

    # parse the args and the commands
    args = parser.parse_args()
    apply_pipeline, cmds = pipeline.parse_and_build(args.commands)

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger('spec-enhance')

    # Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand the input list
    input_images = sorted(expand_path_list(args.input_images))
    if len(input_images) == 0:
        logger.warning('Nothing to process')
        sys.exit()

    # Write metadata log
    if args.logfile:
        now_ts = dt.now(tz.utc)
        meta = {
            'program': f'{parser.prog} v{ENHANCE_VERSION}',
            'created': now_ts.strftime('%m/%d/%Y, %H:%M:%S (%Z)'),
            'args': args.__dict__,
            'images': [str(i) for i in input_images],
            'pipeline': cmds
        }
        now_str = now_ts.strftime('%Y%m%d_%H%M%S')
        meta_file = out_dir / f'{now_str}_{parser.prog}.json'
        with meta_file.open('w') as of:
            json.dump(meta, of, indent=2)

    # Iterate the images
    for img_path in tqdm(input_images, desc=f'Enhancing images',
                         disable=not args.progress):
        # Check if file exists
        if not img_path.exists():
            logger.error(f'File does not exist: {img_path}')
            continue

        # Load image
        try:
            img = iio.imread(img_path)
        except ValueError as e:
            logger.error(f'Failed to load file: {str(img_path)}')
            logger.debug(exc_info=e)
            continue
        in_dtype = img.dtype

        # Convert to float for processing
        img = img_as_float(img)

        # Process the image
        img = apply_pipeline(img)

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
        img = imgproc.as_dtype(img, out_dtype)

        # Format specific opts
        q = args.quality
        if out_fmt in ['.jpg', '.jpeg']:
            kwargs['quality'] = 100 if q is None else q
        elif out_fmt in ['.tif', '.tiff']:
            kwargs['compression'] = 'zlib'
            kwargs['compressionargs'] = {'level': 9 if q is None else q}

        # Save the image to disk
        out_file = f'{img_path.stem}{args.suffix}{out_fmt}'
        out_path = out_dir / out_file
        iio.imwrite(out_path, img, **kwargs)

        # Copy metadata
        if args.metadata:
            extra_tags = [f'-Software=EduceLab spec-enhance v{ENHANCE_VERSION}']
            exiftool.copy_all(img_path, out_path, extra_tags=extra_tags)


if __name__ == '__main__':
    main()
