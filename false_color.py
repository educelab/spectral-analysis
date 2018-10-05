import argparse

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('red')
    parser.add_argument('green')
    parser.add_argument('blue')
    parser.add_argument('output')
    args = parser.parse_args()

    # Read in a 32 bit tif
    r = np.asarray(Image.open(args.red)) * 256
    g = np.asarray(Image.open(args.green)) * 256
    b = np.asarray(Image.open(args.blue)) * 256
    # Write an 8 bit tif because I couldn't get 32 bit to work
    o = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
    o[...,0] = r
    o[...,1] = g
    o[...,2] = b
    im = Image.fromarray(o)
    im.save(args.output)


if __name__ == '__main__':
    main()
