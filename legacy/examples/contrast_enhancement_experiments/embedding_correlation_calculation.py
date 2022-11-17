import os
from random import seed, shuffle

import imageio
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn

from legacy.spectral_analysis.io import SpectralDataHandler


class TwoLayerAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TwoLayerAutoEncoder, self).__init__()

        self.encoder0 = nn.Linear(input_dim, 128)
        self.encoder1 = nn.Linear(128, embedding_dim)

        self.decoder0 = nn.Linear(embedding_dim, 128)
        self.decoder1 = nn.Linear(128, input_dim)

    def forward(self, x, embed=False):
        embedding = self.encoder1(nn.functional.sigmoid(self.encoder0(x)))

        if embed:
            return embedding

        else:
            x = self.decoder1(nn.functional.sigmoid(self.decoder0(embedding)))
            return embedding, x


def get_training_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append((path.strip() for path in line.split(",")))

    return file_paths


def calculate_correlation_statistics(df: pd.DataFrame, save_dir, data_source):
    correlation_matrix = df.corr()

    unique_correlations = []

    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[0]):
            if i > j:
                unique_correlations.append(correlation_matrix.values[i, j])

    unique_correlations = correlation_matrix.values[0]

    correlation_matrix.to_csv(os.path.join(save_dir, f"correlation_matrix_{data_source}.csv"))

    average_correlation = unique_correlations.mean()
    standard_deviation_correlation = unique_correlations.std()

    print(f"Mean for {data_source} is {average_correlation}")
    print(f"Stf for {data_source} is {standard_deviation_correlation}")

    # if correlation_matrix.shape[0] > 40:
    #     correlation_matrix = correlation_matrix[
    #         [True if i % 5 == 0 else False for i in range(correlation_matrix.shape[0])]
    #     ]
    #
    #
    # fig = plt.figure()
    #
    # ax = sns.heatmap(
    #     correlation_matrix,
    #     vmin=-1, vmax=1, center=0,
    #     cmap=sns.diverging_palette(20, 220, n=200),
    #     square=True
    # )
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     rotation=45,
    #     horizontalalignment='right'
    # )
    #
    # plt.savefig(os.path.join(save_dir, f"correlation_plot_{data_source}.png"))


if __name__ == '__main__':
    print("Starting Correlation Analysis", flush=True)

    OUTPUT_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/outputs/correlation_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pca_path = "/scratch/dbdo224/2020-hyperspectral/outputs/pca_experiment/fitted_pca.joblib"
    autoencoder_path = "/scratch/dbdo224/2020-hyperspectral/outputs/two_layer_autoencoder_experiment/two_layer_autoencoder.pt"
    DATA_REFERENCE = "/spectral-analysis/examples/contrast_enhancement_experiments/autoencoder_method/data_and_mask_paths.txt"

    print("Loading Training Data", flush=True)

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

    dataset = np.concatenate(training_chunk)

    print(f"Loaded {len(dataset)} number of pixels", flush=True)

    seed(463)
    shuffle(dataset)
    training_data = dataset[:int(np.ceil(0.8 * len(dataset)))]
    validation_data = dataset[int(np.ceil(0.8 * len(dataset))):]

    training_df = pd.DataFrame(training_data, columns=[f"Band_{i}" for i in range(370)])
    validation_df = pd.DataFrame(validation_data, columns=[f"Band_{i}" for i in range(370)])

    # print("Correlating bands for raw training and validation data", flush=True)
    # calculate_correlation_statistics(training_df, OUTPUT_DIR, "training_data")
    # calculate_correlation_statistics(validation_df, OUTPUT_DIR, "validation_data")

    # print("Loading AutoEncoder from file", flush=True)
    # autoencoder = torch.load(autoencoder_path)
    #
    # print("Applying AutoEncoder to Training Data", flush=True)
    # transformed_pixels = np.zeros(shape=(training_data.shape[0], 3))
    # for i, data_point in enumerate(training_data):
    #     data_point = torch.tensor(data_point, dtype=torch.float)
    #     transformed_pixels[i] = autoencoder.forward(data_point, embed=True).detach().numpy()
    #
    # transformed_df = pd.DataFrame(transformed_pixels, columns=[f"Embedding_{i}" for i in range(3)])
    #
    # print("Correlating bands for training data with autoencoder applied", flush=True)
    # calculate_correlation_statistics(transformed_df, OUTPUT_DIR, "autoencoder_training")
    #
    # print("Applying AutoEncoder to Validation Data", flush=True)
    # transformed_pixels = np.zeros(shape=(validation_data.shape[0], 3))
    # for i, data_point in enumerate(validation_data):
    #     data_point = torch.tensor(data_point, dtype=torch.float)
    #     transformed_pixels[i] = autoencoder.forward(data_point, embed=True).detach().numpy()
    #
    # transformed_df = pd.DataFrame(transformed_pixels, columns=[f"Embedding_{i}" for i in range(3)])
    #
    # print("Correlating bands for validation data with autoencoder applied", flush=True)
    # calculate_correlation_statistics(transformed_df, OUTPUT_DIR, "autoencoder_validation")

    print("Loading PCA from file", flush=True)
    pca = joblib.load(pca_path)

    print("Applying PCA to Training Data", flush=True)
    transformed_pixels = np.zeros(shape=(training_data.shape[0], 3))
    for i, data_point in enumerate(training_data):
        transformed_pixels[i] = pca.transform(np.expand_dims(data_point, axis=0))

    transformed_df = pd.DataFrame(transformed_pixels, columns=[f"Embedding_{i}" for i in range(3)])

    print("Correlating bands for training data with pca applied", flush=True)
    calculate_correlation_statistics(transformed_df, OUTPUT_DIR, "pca_training")

    print("Applying PCA to Validation Data", flush=True)
    transformed_pixels = np.zeros(shape=(validation_data.shape[0], 3))
    for i, data_point in enumerate(validation_data):
        transformed_pixels[i] = pca.transform(np.expand_dims(data_point, axis=0))

    transformed_df = pd.DataFrame(transformed_pixels, columns=[f"Embedding_{i}" for i in range(3)])

    print("Correlating bands for validation data with PCA applied", flush=True)
    calculate_correlation_statistics(transformed_df, OUTPUT_DIR, "pca_validation")

    print("Done", flush=True)
