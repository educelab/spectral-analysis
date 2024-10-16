import argparse
import string
import warnings
from os import PathLike
from pathlib import Path
from typing import Union, List

import imageio.v3 as iio
import numpy as np
import zarr
from numcodecs import Blosc

# This script demonstrates converting flatfield-corrected MegaVision 
# datasets into zarr files.

def list_flattened_files(dataset_dir: Union[str, PathLike]) -> List[Path]:
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    flat_dir = dataset_dir / 'Flattened'
    files = list(flat_dir.glob('*_F.tif'))
    files.sort(key=lambda p: int(p.name.split('_')[1]))
    return files


def get_dtype_max(dtype):
    if np.isdtype(dtype, np.uint8):
        return 255
    elif np.isdtype(dtype, np.uint16):
        return 65535
    elif np.isdtype(dtype, np.float32):
        return 1.
    else:
        raise RuntimeError(f'Unsupported dtype: {dtype}')


def get_dtype_bytes(dtype):
    if np.isdtype(dtype, 'integral'):
        return np.iinfo(dtype).bits // 8
    elif np.isdtype(dtype, 'real floating'):
        return np.finfo(dtype).bits // 8
    else:
        raise RuntimeError(f'Unsupported dtype: {dtype}')


# Simple linear scale
def scale_to_dtype(a, dtype):
    if np.isdtype(a.dtype, dtype):
        return a
    in_max = get_dtype_max(a.dtype)
    out_max = get_dtype_max(dtype)
    return (a * (out_max / in_max)).astype(dtype)


def calculate_chunk_size(shape, dtype, chunk_bytes=2_000_000):
    bpp = get_dtype_bytes(dtype)
    num_elems = chunk_bytes // bpp

    # calculate chunk shape
    cn_dim = shape[2] // 2
    xy_elems = num_elems // cn_dim
    xy_dim = int(np.sqrt(xy_elems))
    chunk_shape = (xy_dim, xy_dim, cn_dim)

    return chunk_shape


def parse_wavelength(path) -> int:
    tail = path.name.split('+')[-1]
    wl = tail.split('_')[0]
    wl = ''.join([c for c in wl if c in string.digits])
    return int(wl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-datasets', '-i', metavar='PATH', nargs='+')
    parser.add_argument('--output-dir', '-o', metavar='PATH',
                        help='Output directory for zarr datasets. '
                             'Default: path/to/dataset.zarr')
    parser.add_argument('--dtype', choices=['uint8', 'uint16', 'float32'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # silence imageio warnings
    warnings.filterwarnings('ignore')

    # Get output dir
    output_dir = Path(args.output_dir)

    # Convert each dataset
    for ds in args.input_datasets:
        ds = Path(ds)
        files = list_flattened_files(ds)

        if len(files) != 17:
            print(f'WARNING: Expected 17 images, but found {len(files)}')

        # Drop the last image: 017
        if len(files) == 17:
            files.pop()

        # dataset output dir
        out_dir = output_dir
        if out_dir is None:
            out_dir = ds.resolve().parent
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / (ds.name + '.zarr')

        # Don't overwrite existing
        if out_path.exists():
            print(f'WARNING: zarr dataset already exists: {str(out_path)}')
            if not args.overwrite:
                print(f'Overwrite disabled. Skipping dataset')
                continue

        # Iterate through files
        arr = None
        pad = len(str(len(files)))
        print(f'{ds.name}')
        wavelengths = []
        for i, f in enumerate(files):
            # parse the wavelength from the filename
            wl = parse_wavelength(f)
            wavelengths.append(wl)

            # Load the image
            print(f' - channel {i:>{pad}}: {f.name} ({wl}nm)')
            img = iio.imread(f)

            # (Optional) Change pixel bit depth
            if args.dtype is not None:
                img = scale_to_dtype(img, np.dtype(args.dtype))

            # Create zarr using first image shape
            if arr is None:
                # array shape
                shape = img.shape[:2] + (len(files),)
                # calculate chunk size
                chunk_shape = calculate_chunk_size(shape, img.dtype)
                # open zarr file
                arr = zarr.open(out_path,
                                mode='w',
                                compressor=Blosc(cname='zstd', clevel=1),
                                shape=shape,
                                chunks=chunk_shape,
                                dtype=img.dtype)

            # write image to channel
            arr[..., i] = img
        arr.attrs['wavelengths'] = wavelengths

        # Print a space between datasets
        print()


if __name__ == '__main__':
    main()
