import os
from random import shuffle, seed

import imageio
import numpy as np
import torch
import torch.nn as nn
from spectral_io import SpectralDataHandler


def get_training_data_paths(f_name):
    file_paths = []
    with open(f_name) as infile:
        for line in infile:
            if line.strip() != "":
                file_paths.append((path.strip() for path in line.split(",")))

    return file_paths


class DropoutAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(DropoutAutoencoder, self).__init__()

        self.encoder0 = nn.Linear(input_dim, 256)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout(p=0.3)
        self.encoder1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.encoder2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.encoder3 = nn.Linear(64, embedding_dim)
        self.embedding = nn.Sigmoid()

        self.decoder1 = nn.Linear(embedding_dim, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)
        self.decoder2 = nn.Linear(64, 128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.3)
        self.decoder3 = nn.Linear(128, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.3)
        self.decoder4 = nn.Linear(256, input_dim)

        self.reconstruction = nn.Sigmoid()

    def forward(self, x, embed=False):
        x = self.dropout0(self.relu0(self.encoder0(x)))
        x = self.dropout1(self.relu1(self.encoder1(x)))
        x = self.dropout2(self.relu2(self.encoder2(x)))
        x = self.embedding(self.encoder3(x))

        if embed:
            return x

        else:
            x = self.dropout3(self.relu3(self.decoder1(x)))
            x = self.dropout4(self.relu4(self.decoder2(x)))
            x = self.dropout5(self.relu5(self.decoder3(x)))
            x = self.reconstruction(self.decoder4(x))
            return x


if __name__ == '__main__':
    print("Starting Autoencoder Experiment", flush=True)
    OUTPUT_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/outputs/autoencoder_experiment")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    EMBEDDING_DIM = 3
    MINIBATCH_SIZE = 256
    Autoencoder_FILENAME = os.path.join(OUTPUT_DIR, "fitted_autoencoder.pt")
    DATA_REFERENCE = "/spectral-analysis/examples/contrast_enhancement_experiments/autoencoder_method/data_and_mask_paths.txt"

    FILE_TYPE = "tiff"

    if os.path.exists(Autoencoder_FILENAME):
        print("Loading existing Autoencoder model", flush=True)

    else:
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

        print("Beginning Autoencoder training", flush=True)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on GPU", flush=True)
        else:
            device = torch.device("cpu")
            print("Running on the CPU", flush=True)

        model = DropoutAutoencoder(370, EMBEDDING_DIM).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        running_val_loss = []

        for epoch in range(500):
            model.train()
            shuffle(training_data)
            for i in range(int(np.ceil(len(training_data) / MINIBATCH_SIZE))):
                training_batch = torch.tensor(training_data[MINIBATCH_SIZE * i: MINIBATCH_SIZE * (i + 1)],
                                              dtype=torch.float)
                output_pred = model(training_batch)

                loss = criterion(output_pred, training_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            validation_loss = []
            for i in range(int(np.ceil(len(validation_data) / MINIBATCH_SIZE))):
                validation_batch = torch.tensor(validation_data[MINIBATCH_SIZE * i: MINIBATCH_SIZE * (i + 1)],
                                                dtype=torch.float)
                output_pred = model(validation_batch)

                validation_loss.append(criterion(output_pred, validation_batch).item())

            running_val_loss.append(np.mean(validation_loss))

            print(f"Validation loss for epoch {epoch} == {running_val_loss[-1]}", flush=True)

            if not np.any([running_val_loss[-1] < running_val_loss[-10: -1]]) and epoch > 1:
                print(f"Validation loss plateau, stopping training at epoch {epoch}", flush=True)
                break

        torch.save(model, Autoencoder_FILENAME)

    print("Loading Autoencoder from file", flush=True)
    autoencoder = torch.load(Autoencoder_FILENAME)
    autoencoder.eval()

    for data_file, _ in get_training_data_paths(DATA_REFERENCE):
        print(f"Applying transform to {data_file}", flush=True)
        data_handler = SpectralDataHandler(data_file)
        x_max = data_handler.io.metadata["samples"]
        y_max = data_handler.io.metadata["lines"]
        z_max = data_handler.io.metadata["bands"]

        image = np.zeros((y_max, x_max, EMBEDDING_DIM))

        for i in range(y_max):
            fitting_data = data_handler.io.get_volume_chunk((0, x_max), (i, i + 1), (0, z_max)).reshape((-1, z_max))
            fitting_data = torch.tensor(fitting_data, dtype=torch.float)
            image[i] = autoencoder.forward(fitting_data, embed=True).detach().numpy()

        for i in range(EMBEDDING_DIM):
            file_name = data_file.split("/")[-1].split(".")[0]
            imageio.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_dimension.{FILE_TYPE}"), image[:, :, i])

        if EMBEDDING_DIM == 3:
            file_name = data_file.split("/")[-1].split(".")[0]
            imageio.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_false_color.{FILE_TYPE}"), image)

    print("Done", flush=True)
