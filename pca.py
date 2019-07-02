import argparse
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sklearn.decomposition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='input hdf5')
    # parser.add_argument('output', help='output hdf5 file name')
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as f:
        dset = f['image']
        width, height = dset.shape[1], dset.shape[0]

        # plt.plot(dset[1000][1000])
        # plt.show()

        randoms = []

        for i in range(10000):
            if i % 1000 == 0:
                print(i)
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 1)
            randoms.append(dset[y][x])

        n_components = 10
        pca = sklearn.decomposition.PCA(n_components=n_components)
        pca.fit(randoms)
        # print(pca.transform(randoms)[0:10])

        step_size = 10
        for component in range(n_components):
            values = np.zeros((height, width), dtype=np.float32)
            for y in range(0, height, step_size):
                for x in range(0, width, step_size):
                    values[y-(step_size//2):y+(step_size//2), x-(step_size//2):x+(step_size//2)] = pca.transform([dset[y][x]])[0][component]
            values -= values.min()
            values /= values.max()
            values *= 256
            im = Image.fromarray(values)
            im.save(str(component) + '.tif')

if __name__ == '__main__':
    main()
