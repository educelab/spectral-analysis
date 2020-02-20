import os

import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spectral_data_io import SpectralDataHandler


def get_training_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append((path.strip() for path in line.split(",")))

    return file_paths


if __name__ == '__main__':
    NUM_BINS = 200
    PROJECT_DIR = "../test_runs/PCA_test_masked_characters_floats/"
    DATA_MASK_REFERENCE_FILE = PROJECT_DIR + "data_source.txt"
    CANONICAL_INK_PIXEL_PATH = "/Volumes/HD-Daniel/PHerc118/Photos/2017-Hyperspectral/RawScans/PHerc118-Pezzo1/2017_07_17_12_08_27/2017_07_17_12_08_27data.hdr"
    CANONICAL_INK_PIXEL_COORDINATES = (1274, 3847)

    training_chunk = []
    for data_file, mask_file in get_training_data_paths(DATA_MASK_REFERENCE_FILE):
        data_handler = SpectralDataHandler(data_file)
        x_max = data_handler.io.metadata["samples"]
        y_max = data_handler.io.metadata["lines"]
        z_max = data_handler.io.metadata["bands"]

        if os.path.exists(mask_file):
            mask = imageio.imread(mask_file)[:, :, 3]
            assert mask.shape == (y_max, x_max)
        else:
            mask = np.ones((y_max, x_max))

        for i in tqdm(range(y_max)):
            if np.any(mask[i] != 0):
                training_data = data_handler.io.get_volume_chunk(
                    (0, x_max), (i, i + 1), (0, z_max)
                ).reshape((-1, z_max))

                training_data = training_data[mask[i] != 0]
                training_chunk.append(training_data)

    training_data = np.concatenate(training_chunk)

    data_handler = SpectralDataHandler(CANONICAL_INK_PIXEL_PATH)
    canonical_pixel_bands = data_handler.io.get_volume_chunk((CANONICAL_INK_PIXEL_COORDINATES[0],
                                                              CANONICAL_INK_PIXEL_COORDINATES[0] + 1),
                                                             (CANONICAL_INK_PIXEL_COORDINATES[1],
                                                              CANONICAL_INK_PIXEL_COORDINATES[1] + 1),
                                                             (0, data_handler.io.metadata["bands"])
                                                             ).flatten()

    for i in range(training_data.shape[1]):
        fig = plt.figure()
        N, bins, patches = plt.hist(training_data[:, i], bins=NUM_BINS)
        plt.title(f"Pixel intensity vs. count across band {i + 1}")
        plt.xlabel("Pixel intensity")
        plt.ylabel(f"Count per intensity bin with {NUM_BINS} bins")

        for j in range(len(bins)):
            if j != len(bins) and bins[j] <= canonical_pixel_bands[i] < bins[j + 1]:
                patches[j].set_facecolor(plt.cm.viridis(1))

        plt.savefig(os.path.join(PROJECT_DIR, "training_data_analysis", f"intensity_distribution_band{i + 1}_with_ink_response.jpg"))

    fig = plt.figure()
    plt.hist(training_data[:, 0])

    def update_hist(num, data):
        plt.cla()
        N, bins, patches = plt.hist(data[:, num], bins=NUM_BINS)
        plt.title(f"Pixel intensity vs. count across band {num + 1}")
        plt.xlabel("Pixel intensity")
        plt.ylabel(f"Count per intensity bin with {NUM_BINS} bins")

        for k in range(len(bins)):
            if k != len(bins) and bins[k] <= canonical_pixel_bands[num] < bins[k + 1]:
                patches[k].set_facecolor(plt.cm.viridis(1))

    animation = animation.FuncAnimation(fig, update_hist, training_data.shape[1], fargs=(training_data,))
    animation.save(os.path.join(PROJECT_DIR, "training_data_analysis", f"intensity_distribution_animation_with_ink_response.gif"))
    plt.show()

