import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from struct import unpack_from
from typing import Tuple

import numpy as np


class AbstractSpectralIO(ABC):
    file_path: str = None
    metadata: dict = None

    @abstractmethod
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        pass

    @abstractmethod
    def get_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int]) -> np.ndarray:
        pass

    @abstractmethod
    def set_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int],
                         data: np.ndarray) -> None:
        pass

    def get_metadata(self) -> dict:
        return self.metadata

    # TODO support masked dataset loading


class ENVISpectralIO(AbstractSpectralIO):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.metadata = self._validate_envi_header(self.file_path)

    # TODO implement __getitem__ to support slicing operations

    def get_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int]) -> np.ndarray:

        # Validates range coordinates
        x_range = (max(0, x_range[0]), min(self.metadata["samples"], x_range[1]))
        y_range = (max(0, y_range[0]), min(self.metadata["lines"], y_range[1]))
        z_range = (max(0, z_range[0]), min(self.metadata["bands"], z_range[1]))

        if self.metadata["byte order"] == 0:
            byte_order_flag = "<"
        else:
            byte_order_flag = ">"

        if self.metadata["data type"] == 1:
            num_bytes_per_data_point = 1
            format_character_flag = "B"
            data_type = np.uint8

        elif self.metadata["data type"] == 2:
            num_bytes_per_data_point = 2
            format_character_flag = "h"
            data_type = np.int16

        elif self.metadata["data type"] == 3:
            num_bytes_per_data_point = 4
            format_character_flag = "i"
            data_type = np.int32

        elif self.metadata["data type"] == 4:
            num_bytes_per_data_point = 4
            format_character_flag = "f"
            data_type = np.float32

        elif self.metadata["data type"] == 5:
            num_bytes_per_data_point = 8
            format_character_flag = "d"
            data_type = np.float64

        elif self.metadata["data type"] == 6:
            raise NotImplementedError("Error, this parser does not yet support complex valued data")  # TODO

        elif self.metadata["data type"] == 9:
            raise NotImplementedError("Error, this parser does not yet support complex valued data")  # TODO

        elif self.metadata["data type"] == 12:
            num_bytes_per_data_point = 2
            format_character_flag = "H"
            data_type = np.uint16

        elif self.metadata["data type"] == 13:
            num_bytes_per_data_point = 4
            format_character_flag = "I"
            data_type = np.uint32

        elif self.metadata["data type"] == 14:
            num_bytes_per_data_point = 8
            format_character_flag = "q"
            data_type = np.int64

        elif self.metadata["data type"] == 15:
            num_bytes_per_data_point = 8
            format_character_flag = "Q"
            data_type = np.uint64

        else:
            raise ValueError(f"Error: data type of value {self.metadata['data type']} un-parsable.")

        volume = np.zeros(shape=(y_range[1]-y_range[0],
                                 x_range[1]-x_range[0],
                                 z_range[1]-z_range[0]), dtype=data_type)

        # TODO, file open persistence using with keyword

        with open(self.file_path.rstrip(".hdr"), "rb") as infile:
            file_offset = (
                    self.metadata["header offset"]
                    + ((x_range[0])
                       + (z_range[0] * self.metadata["samples"])
                       + (y_range[0] * self.metadata["samples"] * self.metadata["bands"]))
                    * num_bytes_per_data_point)

            if self.metadata["interleave"] == "bil":
                for y in range(y_range[1] - y_range[0]):
                    for z in range(z_range[1] - z_range[0]):
                        infile.seek(file_offset, 1)

                        sample = infile.read(num_bytes_per_data_point * (x_range[1] - x_range[0]))

                        converted_sample = unpack_from(byte_order_flag
                                                       + (format_character_flag * (x_range[1] - x_range[0])),
                                                       sample)

                        volume[y, :, z] = converted_sample

                        file_offset = (self.metadata["samples"] - x_range[1] + x_range[0]) * num_bytes_per_data_point
                    file_offset += (self.metadata["samples"]
                                    * (self.metadata["bands"]
                                       - z_range[1]
                                       + z_range[0])) * num_bytes_per_data_point

            elif self.metadata["interleave"] == "bsq":
                raise NotImplementedError("Error, BSQ not yet supported")  # TODO

            elif self.metadata["interleave"] == "bip":
                raise NotImplementedError("Error, BIP not yet supported")  # TODO

            else:
                raise ValueError(f"Error, unable to read interleave type {self.metadata['interleave']}")

        return volume

    def set_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int],
                         data: np.ndarray) -> None:
        raise NotImplementedError

    @staticmethod
    def _parse_envi_header(header_path: str) -> dict:
        header_data = {}
        with open(header_path) as infile:
            data_string = infile.read()

        # Assure proper formatting and remove first line that always says ENVI
        assert data_string.startswith("ENVI\n")
        data_string = data_string[5:]

        # Separate comments from data
        comments = []
        for match in re.finditer(r";(.*)\n?", data_string):
            comments.append(match.group(1))
            data_string = data_string.replace(match.group(0), "", 1)

        if comments:
            header_data["comments"] = comments
            # TODO log that the file had comments and list them

        # Get the required ENVI file metadata
        required_expressions_tuple = [
            ("file type", r"(file type) = (.*)\n?", str),
            ("interleave", r"(interleave) = (bil|bsq|bip)\n?", str),
            ("data type", r"(data type) = (12|13|14|15|1|2|3|4|5|6|9|)\n?", int),
            ("bands", r"(bands) = (\d+)\n?", int),
            ("byte order", r"(byte order) = ([01])\n?", int),
            ("header offset", r"(header offset) = (\d+)\n?", int),
            ("lines", r"(lines) = (\d+)\n?", int),
            ("samples", r"(samples) = (\d+)\n?", int)
        ]
        for keyword, expression, casting_function in required_expressions_tuple:
            # TODO, find a clean way of checking for repeated keywords and warning/throwing an error if that occurs
            match = re.search(expression, data_string)
            if match:
                header_data[keyword] = casting_function(match.group(2))
                data_string = data_string[:match.start()] + data_string[match.end():]
            else:
                raise ValueError(f"Error, the ENVI header is missing the required information {keyword}")

        extra_data = {}
        while len(data_string) > 0:
            match = re.match(r"(.*) = ({(.|\n)*?})\n?", data_string)
            if match:
                extra_data[match.group(1)] = match.group(2)
                data_string = data_string[:match.start()] + data_string[match.end():]
            else:
                match = re.match(r"(.*) = (.*)\n?", data_string)
                if match:
                    extra_data[match.group(1)] = match.group(2)
                    data_string = data_string[:match.start()] + data_string[match.end():]
                else:
                    raise ValueError(f"Can not parse data left in ENVI file: {data_string}")

        if extra_data:
            # TODO, log extra data
            header_data["extra data"] = extra_data

        return header_data

    def _validate_envi_header(self, header_path: str) -> dict:
        try:
            return self._parse_envi_header(header_path)

        except Exception as error:
            print(f"Error, failed to parse ENVI header file {header_path}")
            raise error


class SpectralDataHandler:
    file_path: str = None
    file_format: str = None
    metadata: dict = None

    def __init__(self, file_path: str = None, file_format: str = None):
        if file_path:
            self.file_path = file_path

        if file_format:
            self.file_format = file_format
        else:
            self.file_format = self._infer_file_format(self.file_path)

        if self.file_format == "envi":
            self.io = ENVISpectralIO(self.file_path)
            self.metadata = self.io.get_metadata()

        else:
            # TODO
            raise NotImplementedError(f"Error, file handling for file format {self.file_format} not yet supported")

    def _infer_file_format(self, file_path: str) -> str:
        if file_path.endswith((".hdf", ".hdf5", ".h5", "he5")):
            return "hdf"
        elif file_path.endswith((".fits", ".fit", ".fts")):
            return "fits"
        elif file_path.endswith(".hdr") and os.path.exists(file_path[:-4]):
            return "envi"
        elif "." not in file_path and os.path.exists(file_path + ".hdr"):
            self.file_path = file_path + ".hdr"  # TODO, not sure if we should allow inputting data file or header
            return "envi"
        elif os.path.isdir(file_path):
            # TODO, check that format matches spectral camera output
            pass
        else:
            raise ValueError(f"Error, could not infer file format for file {file_path}. Is it supported?")


class SpectralPackageManager:
    def __init__(self, project_path, project_id):
        self.filepath = os.path.join(project_path, project_id + '.spectralpkg')

        if os.path.exists(self.filepath):
            # If project already exists just start logger
            logging.basicConfig(filename=os.path.join(self.filepath, 'Logging.log'))

        else:
            # TODO figure out why logger isn't logging
            # If project doesn't exist make the project and start the logger
            os.mkdir(self.filepath)
            logging.basicConfig(filename=os.path.join(self.filepath, 'Logging.log'))
            logging.info("SpectralPkg named {} created at {}".format(project_id + '.spectralpkg', project_path))

            for top_level_folder in ("Artifacts", "Transforms", "Analysis"):
                os.makedirs(os.path.join(self.filepath, top_level_folder))
            logging.info("Top level directories successfully created")

    def add_sample(self, path, overwrite=False, use_sym=False, art_id=None):
        if os.path.isdir(path):
            if art_id is None:
                art_id = os.path.basename(os.path.normpath(path))
            filetype = ""

        else:
            art_id_temp, filetype = os.path.basename(path).split(".")
            if art_id is None:
                art_id = art_id_temp

        # Check to see if the destination folder already exists
        destination = os.path.join(self.filepath, "Artifacts", art_id)

        if os.path.exists(destination) and not overwrite:
            logging.warning('Tried to write file {} to {}. '
                            'Process stopped due to existing file at that location'.format(path, destination))
            raise ValueError("Error, artifact with that file name already exists")

        os.makedirs(os.path.join(destination, 'SpectralData'), exist_ok=overwrite)

        # TODO, change this to URI style system (save original path)
        if use_sym:
            os.symlink(path, os.path.join(destination, 'SpectralData', 'spectraldata' + filetype),
                       target_is_directory=os.path.isdir(path))
        else:
            shutil.copytree(path, os.path.join(destination, 'SpectralData', 'spectraldata' + filetype))
