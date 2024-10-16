import argparse
from pathlib import Path

import imageio.v3 as iio
import zarr

# This script demonstrates dumping the channels of a zarr dataset to TIFF files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True, help='Zarr path')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--path', default='')
    args = parser.parse_args()

    # load input file
    input_path = Path(args.input_file)
    arr = zarr.open(input_path, mode='r', path=args.path)

    # set up output directory
    output_dir = Path()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    output_dir.resolve().mkdir(exist_ok=True, parents=True)

    # set up output files
    stem = input_path.name.removesuffix(''.join(input_path.suffixes))

    num_files = arr.shape[-1]
    pad = len(str(num_files))
    for i in range(num_files):
        print(f'Dumping channel {i:>{pad}}')
        img = arr[..., i]
        suffix = ''
        if 'wavelengths' in arr.attrs.keys():
            wl = arr.attrs['wavelengths'][i]
            suffix = f'_{wl:04}nm'
        out_path = output_dir / f'{stem}_{i:0{pad}}{suffix}.tif'
        iio.imwrite(out_path, img)


if __name__ == '__main__':
    main()
