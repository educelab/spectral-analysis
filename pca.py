import argparse
import os
import random

import h5py
from PIL import Image
import sklearn.decomposition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='input hdf5')
    # parser.add_argument('output', help='output hdf5 file name')
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as f:
        dset = f['points']

        print(type(dset))
        pca = sklearn.decomposition.PCA(n_components=10)
        pca.fit(dset)
        print(pca.transform(dset)[0:10])


if __name__ == '__main__':
    main()
