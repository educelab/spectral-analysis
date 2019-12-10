import imageio

from spectral_data_io import SpectralDataHandler

if __name__ == "__main__":
    # TODO write argument parser for file io
    x = SpectralDataHandler("/Volumes/HD-Daniel/PHerc118/Photos/2017-Hyperspectral/"
                            "RawScans/PHerc118-Pezzo1/2017_07_17_10_25_13/2017_07_17_10_25_13raw.hdr")
    for i in range(x.metadata["bands"]):
        print(f"Processing band {i}")
        data = x.io.get_volume_chunk((0, x.metadata["samples"]), (0, x.metadata["lines"]), (i, i + 1))
        imageio.imwrite(f"pezzo_1_bands/{i}.png", data[:, :, 0])
