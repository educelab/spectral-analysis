import argparse
from pathlib import Path

from skimage import (img_as_float, io)
import numpy as np
from educelab import imgproc

from spec_tools.utils.apps import to_numpy_dtype

def main():
    parser = argparse.ArgumentParser('spec-apply-flats')
    parser.add_argument('--input', '-i', metavar='IMAGE', required=True,
                        help='Input image file. Supports 8, 16, and 32-bit '
                             'grayscale images')
    parser.add_argument('--flat', '-f', metavar='FLAT')
    parser.add_argument('--dark', '-d', metavar='DARK')
    parser.add_argument('--output-depth', metavar='INT',
                        type=to_numpy_dtype,
                        help='Specify the preferred output bits-per-channel. '
                             'By default, the output depth matches the input '
                             'depth. Supported depths: 8, 16, 32')
    parser.add_argument('--quality', '-q', type=int,
                        help='Output format-specific compression level')
    parser.add_argument('--output', '-o', metavar='IMAGE', required=True,
                        help='Output image file.')
    args = parser.parse_args()

    img_path = Path(args.input)
    img = io.imread(img_path)

    in_dtype = img.dtype

    # Convert to float for processing
    img = img_as_float(img)

    # Load flats and darks
    if args.flat is None:
        flat = np.ones_like(img)
    else:
        flat = img_as_float(io.imread(args.flat))
        flat[flat == 0.] = 1e-5
    if args.dark is None:
        dark = np.zeros_like(img)
    else:
        dark = img_as_float(io.imread(args.dark))

    # Process the image
    img = imgproc.flatfield_correction(img, flat, dark)

    # Determine output format
    out_path = Path(args.output)
    out_fmt = out_path.suffix
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
    io.imsave(out_path, img, check_contrast=False, **kwargs)

if __name__ == '__main__':
    main()