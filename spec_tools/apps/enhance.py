import argparse
import json
import logging
import sys
import textwrap
from datetime import datetime as dt, timezone as tz
from functools import partial
from pathlib import Path

import numpy as np
from educelab import cmdparse, imgproc
from educelab.imgproc import exiftool
from skimage import (img_as_float, io)
from tqdm import tqdm

from spec_tools.utils.apps import (expand_path_list, setup_logging,
                                   to_numpy_dtype)

VERSION_MAJOR = 1
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_SUFFIX = ''
ENHANCE_VERSION = f'{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}{VERSION_SUFFIX}'

enhance_fns = {
    'clahe': imgproc.clahe,
    'clip': imgproc.clip,
    'curves': imgproc.curves,
    'exposure': imgproc.exposure,
    'gamma': imgproc.gamma_correction,
    'normalize': imgproc.normalize,
    'shadows': imgproc.shadows,
    'sharpen': imgproc.sharpen,
    'stretch': imgproc.stretch,
    'pstretch': imgproc.stretch_percentile
}

class EnhanceParsers:
    @staticmethod
    def clahe(sep, val):
        defaults = {'kernel_size': None, 'nbins': 256}
        parsed = cmdparse.parse_parameter_list(val, ['kernel_size', 'nbins'],
                                               [int, int])
        return defaults | parsed

    @staticmethod
    def clip(sep, val):
        defaults = {'a_min': 0., 'a_max': 1.}
        parsed = cmdparse.parse_parameter_list(val, ['a_min', 'a_max'], [float, float])
        return defaults | parsed

    @staticmethod
    def curves(sep, val):
        if sep != '' or val != '':
            raise ValueError('-curves does not take parameters')
        # TODO: currently a hardcoded enhancement curve
        return {'x': [[0., 0.], [0.207, 0.118], [0.513, 0.473], [1., 1.]]}

    @staticmethod
    def exposure(sep, val):
        defaults = {'val': 1.}
        parsed = cmdparse.parse_parameter_list(val, ['val'], [float])
        return defaults | parsed

    @staticmethod
    def gamma(sep, val):
        defaults = {'gamma': 1., 'gain': 1.}
        parsed = cmdparse.parse_parameter_list(val, ['gamma', 'gain'],
                                               [float, float])
        return defaults | parsed

    @staticmethod
    def normalize(sep, val):
        if sep != '' or val != '':
            raise ValueError('-normalize does not take parameters')
        return {}

    @staticmethod
    def shadows(sep, val):
        return cmdparse.parse_parameter_list(val, ['val'], [float],
                                             num_required=1, mode='+')

    @staticmethod
    def sharpen(sep, val):
        defaults = {'radius': 1., 'amount': 1.}
        parsed = cmdparse.parse_parameter_list(val, ['radius', 'amount'],
                                               [float, float])
        return defaults | parsed

    @staticmethod
    def stretch(sep, val):
        return cmdparse.parse_parameter_list(val, ['a_min', 'a_max'],
                                             [float, float], num_required=2,
                                             mode='+')
    @staticmethod
    def pstretch(sep, val):
        return cmdparse.parse_parameter_list(val, ['min_perc', 'max_perc'],
                                             [float, float], num_required=2,
                                             mode='+')


def build_pipeline(cmd_list):
    pipeline = []
    for (cmd, kwargs) in cmd_list:
        fn = partial(enhance_fns[cmd], **kwargs)
        pipeline.append(fn)
    return pipeline


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
    parser.add_argument('commands', metavar='CMD', nargs='+')

    enhance_opts = parser.add_argument_group('enhancement commands')
    enhance_opts.description = textwrap.dedent("""\
    -clahe{=KERNEL{,BINS}}
    \t\t\tContrast Limited Adaptive Histogram Equalization
    -clip{=MIN{,MAX}}
    \t\t\tClip values to range
    -curves
    \t\t\tCurves enhancement. Currently a preset enhancement curve.
    -exposure{=VAL}
    \t\t\t\Adjust image exposure
    -gamma{=GAMMA{,GAIN}}
    \t\t\tGamma correction
    -normalize
    \t\t\tLinear contrast stretch to data min/max
    -shadows=VAL
    \t\t\tAdjust shadow brightness
    -sharpen{=RADIUS{,AMOUNT}}
    \t\t\tUnsharp masking filter
    -stretch=MIN,MAX
    \t\t\tLinear contrast stretch to absolute values
    -pstretch=MIN,MAX
    \t\t\tLinear contrast stretch to data percentiles
    """)
    # parse the args and the commands
    args = parser.parse_args()
    cmds = cmdparse.parse(args.commands, EnhanceParsers)

    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger('spec-enhance')

    # Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct the processing pipeline
    pipeline = build_pipeline(cmds)

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
            img = io.imread(img_path)
        except ValueError:
            logger.error(f'Failed to load file: {str(img_path)}')
            continue
        in_dtype = img.dtype

        # Convert to float for processing
        img = img_as_float(img)

        # Process the image
        for fn in pipeline:
            img = fn(img)

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
        io.imsave(out_path, img, **kwargs)

        # Copy metadata
        if args.metadata:
            extra_tags = [f'-Software=EduceLab spec-enhance v{ENHANCE_VERSION}']
            exiftool.copy_all(img_path, out_path, extra_tags=extra_tags)


if __name__ == '__main__':
    main()
