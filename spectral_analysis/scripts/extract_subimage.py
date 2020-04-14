# import argparse
import sys
import os

import imageio
import numpy as np

from spectral_analysis.spectral_io import SpectralDataHandler

if __name__ == '__main__':
    # TODO, fix argument parser, modify to support spectral packages
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-o", "--output_dir", help="output directory, defaults to present working directory")
    # parser.add_argument("upper_left", nargs=2, help="upper left point (inclusive)")
    # parser.add_argument("lower_right", nargs=2, help="lower right point (inclusive)")
    # parser.add_argument("files", nargs="+",
    # help="A list of files to have identically placed subimages extracted from")

    # args = parser.parse_args()
    x1, y1, x2, y2 = map(int, sys.argv[1:5])

    output_dir = os.path.abspath(sys.argv[5])
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = map(os.path.abspath, sys.argv[6:-1])
    spectral_band = float(sys.argv[-1])

    OUTPUT_FTYPE = "png"

    for file in files_to_process:
        save_name = f"{os.path.basename(file).split('.')[0]}_cropped.{OUTPUT_FTYPE}"

        if "." in file and not file.endswith(".hdr"):
            image = imageio.imread(file)[y1:y2, x1:x2]
        else:
            data_handler = SpectralDataHandler(file)
            # Do this gross mess because the envi parser counts wavelength metadata as
            # extra info without a dedicated parsing section in the segment it is initially read in
            image_bands = data_handler.metadata["extra data"]["wavelength"].replace(",", "").strip("{}\n").split("\n")
            image_bands = np.array(image_bands, dtype=np.float)
            band_difference = np.abs(image_bands - spectral_band)
            image_index = np.argmin(band_difference)

            print(f"Extracting image with closest wavelength of {image_bands[image_index]}")

            image = data_handler.io.get_volume_chunk((x1, x2), (y1, y2), (image_index, image_index + 1))

        imageio.imsave(os.path.join(output_dir, save_name), image)






