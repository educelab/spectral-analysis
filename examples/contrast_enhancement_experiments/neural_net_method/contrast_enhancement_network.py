import os
from itertools import product
from math import ceil
from random import shuffle

import imageio
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from spectral_analysis.io import SpectralDataHandler


class TestFullConv(nn.Module):
    def __init__(self, number_bands):
        super(TestFullConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=number_bands, out_channels=100, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)


def mse_rms_contrast_metric(x1, x2, alpha=1.0, beta=1.0):
    mse_loss = nn.MSELoss()
    return (mse_loss(x1, x2) * alpha) - (torch.std(x1) * beta)


def mse_kl_regularized_metric(x1, x2, alpha=1.0, beta=1.0):
    mse_loss = nn.MSELoss()

    target_dist = distributions.normal.Normal(torch.mean(x2), beta * torch.std(x2))
    generated_dist = distributions.normal.Normal(torch.mean(x1), torch.std(x1))

    return alpha * mse_loss(x1, x2) + distributions.kl_divergence(target_dist, generated_dist)


def apply_model(model_path, input_path, output_path):
    model = torch.load(model_path)
    data_handler = SpectralDataHandler(input_path)
    image_bands = data_handler.io.metadata["bands"]
    x_coords = data_handler.io.metadata["samples"]
    y_coords = data_handler.io.metadata["lines"]

    output_im = np.zeros((y_coords, x_coords), dtype=float)

    NUM_SUB_IMAGES_X = 10
    NUM_SUB_IMAGES_Y = 10
    sub_image_y_length = ceil(y_coords / NUM_SUB_IMAGES_Y)
    sub_image_x_length = ceil(x_coords / NUM_SUB_IMAGES_X)

    for x_index, y_index in tqdm(list(product(range(NUM_SUB_IMAGES_X), range(NUM_SUB_IMAGES_Y)))):
        image = data_handler.io.get_volume_chunk(
            (x_index * sub_image_x_length, (x_index + 1) * sub_image_x_length),
            (y_index * sub_image_y_length, (y_index + 1) * sub_image_y_length),
            (0, image_bands)
        )
        image = np.moveaxis(image, -1, 0)

        output = model(image).squeeze()

        output_im[y_index * sub_image_y_length: (y_index + 1) * sub_image_y_length,
        x_index * sub_image_x_length: (x_index + 1) * sub_image_x_length] = output.detach().numpy()

    imageio.imwrite(output_path, output_im)


if __name__ == '__main__':
    model_path = "test_models/test_model_dual_loss_kl.pickle"
    LOSS = "MSEKL"

    data_paths = [
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_09_59_43/2017_07_17_09_59_43data.hdr',
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_10_25_13/2017_07_17_10_25_13data.hdr',
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_10_50_43/2017_07_17_10_50_43data.hdr',
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_11_17_19/2017_07_17_11_17_19data.hdr',
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_11_42_49/2017_07_17_11_42_49data.hdr',
        '/Volumes/Samsung_T5/PHerc118-Pezzo1/2017_07_17_12_08_27/2017_07_17_12_08_27data.hdr'
    ]

    if not os.path.exists(model_path):
        model = TestFullConv(370)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        NUM_SUB_IMAGES_X = 10
        NUM_SUB_IMAGES_Y = 10

        for epoch in range(10):
            for path in data_paths:
                data_handler = SpectralDataHandler(path)

                image_bands = data_handler.io.metadata["bands"]
                x_coords = data_handler.io.metadata["samples"]
                y_coords = data_handler.io.metadata["lines"]

                sub_image_y_length = ceil(y_coords / NUM_SUB_IMAGES_Y)
                sub_image_x_length = ceil(x_coords / NUM_SUB_IMAGES_X)

                running_loss = 0

                image_order = list(product(range(NUM_SUB_IMAGES_X), range(NUM_SUB_IMAGES_Y)))
                shuffle(image_order)

                for i, test_image_index in tqdm(enumerate(image_order)):
                    image = data_handler.io.get_volume_chunk(
                        (test_image_index[0] * sub_image_x_length, (test_image_index[0] + 1) * sub_image_x_length),
                        (test_image_index[1] * sub_image_y_length, (test_image_index[1] + 1) * sub_image_y_length),
                        (0, image_bands)
                    )
                    image = np.moveaxis(image, -1, 0)
                    target = torch.tensor(np.mean(image, axis=0))

                    optimizer.zero_grad()

                    output = model(image).squeeze()

                    if LOSS == "MSE":
                        loss = criterion(output, target)
                    elif LOSS == "DualLoss":
                        loss = mse_rms_contrast_metric(output, target)
                    else:
                        loss = mse_kl_regularized_metric(output, target, beta=1.5)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 10 == 9:  # print every 10 iterations
                        print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0

            torch.save(model, model_path)

    for i, path in enumerate(data_paths):
        apply_model(model_path, path, "pezzo1_kl_enhanced_strip{}.tiff".format(i + 1))
