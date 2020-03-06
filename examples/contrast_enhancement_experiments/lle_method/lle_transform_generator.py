import os

import imageio
import numpy as np
from joblib import load, dump
from sklearn.manifold import LocallyLinearEmbedding
from spectral_io import SpectralDataHandler


def get_training_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append((path.strip() for path in line.split(",")))

    return file_paths


if __name__ == '__main__':
    print("Starting LLE Experiment", flush=True)
    OUTPUT_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/outputs/lle_experiment")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    EMBEDDING_DIM = 3
    LLE_FILENAME = os.path.join(OUTPUT_DIR, "fitted_lle.joblib")
    DATA_REFERENCE = "/spectral-analysis/examples/contrast_enhancement_experiments/lle_method/data_and_mask_paths.txt"

    FILE_TYPE = "tiff"

    if os.path.exists(LLE_FILENAME):
        print("Loading existing LLE model", flush=True)
        lle = load(LLE_FILENAME)
    else:
        print("Beginning LLE Training", flush=True)
        lle = LocallyLinearEmbedding(n_components=EMBEDDING_DIM, n_jobs=-1)

        training_chunk = []
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

            for i in range(y_max):
                if np.any(mask[i] != 0):
                    training_data = data_handler.io.get_volume_chunk(
                        (0, x_max), (i, i + 1), (0, z_max)
                    ).reshape((-1, z_max))

                    training_data = training_data[mask[i] != 0]
                    training_chunk.append(training_data)

        training_set = np.concatenate(training_chunk)
        lle.fit(training_set)
        print(f"Trained on {len(training_set)} number of pixels", flush=True)

        dump(lle, LLE_FILENAME)

    print(f"LLE results: Reconstruction error for embedding: {lle.reconstruction_error_}", flush=True)
    for data_file, _ in get_training_data_paths(DATA_REFERENCE):
        print(f"Applying transform to {data_file}", flush=True)
        data_handler = SpectralDataHandler(data_file)
        x_max = data_handler.io.metadata["samples"]
        y_max = data_handler.io.metadata["lines"]
        z_max = data_handler.io.metadata["bands"]

        image = np.zeros((y_max, x_max, EMBEDDING_DIM))

        for i in range(y_max):
            fitting_data = data_handler.io.get_volume_chunk((0, x_max), (i, i + 1), (0, z_max)).reshape((-1, z_max))
            image[i] = lle.transform(fitting_data)

        for i in range(EMBEDDING_DIM):
            file_name = data_file.split("/")[-1].split(".")[0]
            imageio.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_dimension.{FILE_TYPE}"), image[:, :, i])

    print("Done", flush=True)
