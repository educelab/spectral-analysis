import os
import joblib

from tqdm import tqdm
import numpy as np
import imageio

from spectral_analysis.io import SpectralDataHandler


def get_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append(line.strip())

    return file_paths


if __name__ == '__main__':
    NUM_COMPONENTS = 10
    BASE = "/Volumes/HD-Daniel/"
    PROJECT_DIR = os.path.join(BASE, "PHerc118_ExploratoryPCA/")
    PCA_FILENAME = os.path.join(PROJECT_DIR, "TransformAnalysis/fitted_pca.joblib")
    DATA_REFERENCE_FILE = os.path.join(PROJECT_DIR, "data_source.txt")
    FILE_TYPE = "tiff"

    pca = joblib.load(PCA_FILENAME)

    for data_file in get_data_paths(DATA_REFERENCE_FILE):
        data_handler = SpectralDataHandler(data_file)
        x_max = data_handler.io.metadata["samples"]
        y_max = data_handler.io.metadata["lines"]
        z_max = data_handler.io.metadata["bands"]

        image = np.zeros((y_max, x_max, NUM_COMPONENTS))

        for i in tqdm(range(y_max)):
            fitting_data = data_handler.io.get_volume_chunk((0, x_max), (i, i + 1), (0, z_max)).reshape((-1, z_max))
            image[i] = pca.transform(fitting_data)

        for i in range(NUM_COMPONENTS):
            file_directory = os.path.join(PROJECT_DIR, "/".join(data_file[len(BASE):].split("/")[:-1]))
            if not os.path.exists(file_directory):
                os.makedirs(file_directory, exist_ok=True)
            file_name = f"component{i + 1}.{FILE_TYPE}"
            imageio.imwrite(f"{os.path.join(file_directory, file_name)}", image[:, :, i])
