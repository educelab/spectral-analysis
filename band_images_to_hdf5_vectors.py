import argparse
import os
import random
import re

import h5py
import numpy as np
from PIL import Image


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_key(s):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', s)]


def main():
    random.seed(42)
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='input directory of spectral band .tifs')
    parser.add_argument('output', help='output hdf5 file name')
    parser.add_argument('-n', help='number of points to sample', metavar='n', default=100000)
    parser.add_argument('--all-points', action='store_true', default=False)
    args = parser.parse_args()

    files = os.listdir(args.input)
    tifs = sorted([os.path.join(args.input, f) for f in files if f[-4:] == '.tif'], key=natural_key)

    width, height = Image.open(tifs[0]).size

    # if args.all_points:
    #     points = []
    #     for x in range(width):
    #         for y in range(height):
    #             points.append((x, y))
    #     args.n = width * height
    # else:
    #     points = [(random.randint(0, width-1), random.randint(0, height-1)) for n in range(args.n)]

    # with h5py.File(args.output, "w") as f:
    #     dset = f.create_dataset("points", (args.n, len(tifs)), dtype='f')

    #     for t in range(len(tifs)):
    #         print(tifs[t])
    #         im = Image.open(tifs[t])
    #         for p in range(len(points)):
    #             f = im.getpixel(points[p])
    #             dset[p, t] = f

    if args.all_points:
        with h5py.File(args.output, 'w') as f:
            dset = f.create_dataset('image', (height, width, len(tifs)), dtype='f')

            for tif_idx in range(len(tifs)):
                print(tifs[tif_idx])
                im = Image.open(tifs[tif_idx])
                dset[:,:,tif_idx] = np.array(im)

if __name__ == '__main__':
    main()
