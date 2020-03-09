import os

import imageio
import numpy as np
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA, PCA
from spectral_io import SpectralDataHandler
from tqdm import tqdm


def get_training_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append((path.strip() for path in line.split(",")))

    return file_paths


if __name__ == '__main__':
    print("Starting PCA Experiment", flush=True)

    OUTPUT_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/outputs/autoencoder_experiment")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    EMBEDDING_DIM = 3
    PCA_FILENAME = os.path.join(OUTPUT_DIR, "fitted_pca.joblib")
    METHOD = "PCA"
    DATA_REFERENCE = "/spectral-analysis/examples/contrast_enhancement_experiments/autoencoder_method/data_and_mask_paths.txt"
    FILE_TYPE = "tiff"
    # TODO, make the number of y slices pulled each time adaptive to ingest more data at once
    # TODO, figure out this banal multiprocessing for training and pulling data concurrently
    # TODO, move mask handling into the data handler itself for efficiency
    # TODO, add iterative scan (x/z), (x/y), etc. to AbstractSpectralIO

    if os.path.exists(PCA_FILENAME):
        pca = load(PCA_FILENAME)
    else:
        if METHOD == "IterativePCA":
            pca = IncrementalPCA(n_components=EMBEDDING_DIM, whiten=True)

        elif METHOD == "PCA":
            pca = PCA(n_components=EMBEDDING_DIM, whiten=True)

        else:
            raise ValueError(f"{METHOD} is an invalid PCA flavour")

        training_chunk = []
        num_examples = 0
        for data_file, mask_file in get_training_data_paths(DATA_REFERENCE):
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
                    num_examples += len(training_data)

                    if METHOD == "IterativePCA" and (i == y_max - 1 or num_examples >= 10000):
                        pca.partial_fit(np.concatenate(training_chunk))
                        training_chunk = []
                        num_examples = 0

        print("Done loading data", flush=True)
        if METHOD == "PCA":
            pca.fit(np.concatenate(training_chunk))
            print(f"Trained on {num_examples} number of pixels", flush=True)

        dump(pca, PCA_FILENAME)

    for data_file, _ in get_training_data_paths(DATA_REFERENCE):
        print(f"Applying transform to {data_file}", flush=True)
        data_handler = SpectralDataHandler(data_file)
        x_max = data_handler.io.metadata["samples"]
        y_max = data_handler.io.metadata["lines"]
        z_max = data_handler.io.metadata["bands"]

        image = np.zeros((y_max, x_max, EMBEDDING_DIM))

        for i in range(y_max):
            fitting_data = data_handler.io.get_volume_chunk((0, x_max), (i, i + 1), (0, z_max)).reshape((-1, z_max))
            image[i] = pca.transform(fitting_data)

        for i in range(EMBEDDING_DIM):
            file_name = data_file.split("/")[-1].split(".")[0]
            imageio.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_dimension{i}.{FILE_TYPE}"), image[:, :, i])

        if EMBEDDING_DIM == 3:
            file_name = data_file.split("/")[-1].split(".")[0]
            imageio.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_false_color.{FILE_TYPE}"), image)

    print("Done")
